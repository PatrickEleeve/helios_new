# helios/architecture.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from .adapters import EdgeAdapter
from .agents import HFCoreWrapper, EmbeddingNode, AgentNode, LMHeadNode
from .protocols import Role, RoleConfig, FlowConfig, DenseMode

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"

class HeliosArchitecture(nn.Module):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device_map: str | dict | None = "auto",
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        trust_remote_code: bool = True,
        flow: Optional[FlowConfig] = None,
        gradient_checkpointing: bool = False,
        edge_nhead: int = 8,
        edge_ffn_mult: int = 4,
        edge_dropout: float = 0.0,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
        )
        self.core = HFCoreWrapper(self.base, use_gradient_checkpointing=gradient_checkpointing)
        self.h = self.core.hidden_size

        self.flow = flow or FlowConfig()
        self.use_edges = bool(self.flow.use_edges)
        self.dense_mode = self.flow.dense_mode

        # ---- 节点 ----
        self.src = EmbeddingNode(self.core)

        def _mk_role(rc: Role) -> AgentNode:
            return AgentNode(self.core)

        # Analysts 4
        self.roles: Dict[Role, AgentNode] = {}
        for r in self.flow.analyst_roles():
            self.roles[r] = _mk_role(r)

        # Researchers 2
        for r in self.flow.researcher_roles():
            self.roles[r] = _mk_role(r)

        # Trader 1
        self.roles[Role.TRADER] = _mk_role(Role.TRADER)

        # Risk 3
        for r in self.flow.risk_roles():
            self.roles[r] = _mk_role(r)

        # Fund Manager
        self.roles[Role.FUND_MANAGER] = _mk_role(Role.FUND_MANAGER)

        # 输出头
        self.decision_norm = nn.LayerNorm(self.h)
        self.dst = LMHeadNode(self.core)

        # ---- 边（EdgeAdapter）----
        def EA(): return EdgeAdapter(self.h, self.h, nhead=edge_nhead, ffn_mult=edge_ffn_mult, dropout=edge_dropout)

        # src -> Analysts
        self.edge_src_to_analyst = nn.ModuleDict({r.value: EA() for r in self.flow.analyst_roles()})

        # Analysts -> Researchers (共享一条聚合边或各自一条；此处各自一条，再在接收端求和)
        self.edge_analyst_to_bull = nn.ModuleDict({r.value: EA() for r in self.flow.analyst_roles()})
        self.edge_analyst_to_bear = nn.ModuleDict({r.value: EA() for r in self.flow.analyst_roles()})

        # Researchers 双向
        self.edge_bull_to_bear = EA()
        self.edge_bear_to_bull = EA()

        # Researchers -> Trader
        self.edge_bull_to_trader = EA()
        self.edge_bear_to_trader = EA()
        self.edge_analyst_to_trader = nn.ModuleDict({r.value: EA() for r in self.flow.analyst_roles()})

        # Trader -> Risk (三路)
        self.edge_trader_to_risk = nn.ModuleDict({r.value: EA() for r in self.flow.risk_roles()})

        # Risk 三方全连接（不含自环）
        self.edge_risk_pair = nn.ModuleDict()
        risk_list = self.flow.risk_roles()
        for i in range(len(risk_list)):
            for j in range(len(risk_list)):
                if i == j: continue
                self.edge_risk_pair[f"{risk_list[i].value}->{risk_list[j].value}"] = EA()

        # Trader & Risk -> FundManager
        self.edge_trader_to_fm = EA()
        self.edge_riskmix_to_fm = EA()

        # ---- 主持门控：研究员 2 类 + 风控 3 类 ----
        self.research_gate = nn.Linear(self.h, 2, bias=False)  # [bull,bear]
        self.risk_gate = nn.Linear(self.h, 3, bias=False)      # [risky,neutral,safe]

        self._place_custom_modules()

    # 设备/精度对齐
    def _place_custom_modules(self):
        dev = self.core.embed_tokens.weight.device
        dt = self.core.embed_tokens.weight.dtype
        self.src.to(device=dev, dtype=dt)
        self.decision_norm.to(device=dev, dtype=dt)
        for m in self.roles.values(): m.to(device=dev, dtype=dt)
        for md in [
            self.edge_src_to_analyst, self.edge_analyst_to_bull, self.edge_analyst_to_bear,
            self.edge_bull_to_bear, self.edge_bear_to_bull, self.edge_bull_to_trader, self.edge_bear_to_trader,
            self.edge_analyst_to_trader, self.edge_trader_to_risk, self.edge_risk_pair,
            self.edge_trader_to_fm, self.edge_riskmix_to_fm, self.research_gate, self.risk_gate
        ]:
            md.to(device=dev, dtype=dt)

    # 训练阶段控制（与 trainer 接口一致）
    def freeze_vertices(self):
        for p in self.core.parameters(): p.requires_grad_(False)
        for p in self.decision_norm.parameters(): p.requires_grad_(False)

    def unfreeze_vertices(self):
        for p in self.core.parameters(): p.requires_grad_(True)
        for p in self.decision_norm.parameters(): p.requires_grad_(True)

    def edges_parameters(self):
        for md in [
            self.edge_src_to_analyst, self.edge_analyst_to_bull, self.edge_analyst_to_bear,
            self.edge_bull_to_bear, self.edge_bear_to_bull, self.edge_bull_to_trader, self.edge_bear_to_trader,
            self.edge_analyst_to_trader, self.edge_trader_to_risk, self.edge_risk_pair,
            self.edge_trader_to_fm, self.edge_riskmix_to_fm, self.research_gate, self.risk_gate
        ]:
            for p in md.parameters():
                if p.requires_grad: yield p

    def vertices_parameters(self):
        for p in self.core.parameters():
            if p.requires_grad: yield p
        for p in self.decision_norm.parameters():
            if p.requires_grad: yield p

    # --- 前向 ---
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:

        # src
        h_src = self.src(input_ids, attention_mask=attention_mask, position_ids=position_ids)  # [B,T,C]

        # Analysts: src -> each analyst -> node
        h_analysts: Dict[Role, torch.Tensor] = {}
        for r in self.flow.analyst_roles():
            h0 = self.edge_src_to_analyst[r.value](h_src, attention_mask) if self.use_edges else h_src
            h_analysts[r] = self.roles[r](h0, attention_mask=attention_mask, position_ids=position_ids)

        # 聚合分析师态（求和）
        h_analyst_sum = torch.stack(list(h_analysts.values()), dim=0).sum(dim=0)

        # Researchers 初始化：Analyst -> Bull/Bear
        def agg_from(md: nn.ModuleDict) -> torch.Tensor:
            hs = []
            for r in self.flow.analyst_roles():
                hs.append(md[r.value](h_analyst_sum, attention_mask) if self.use_edges else h_analyst_sum)
            return torch.stack(hs, 0).sum(0)

        h_bull = self.roles[Role.RESEARCH_BULL](agg_from(self.edge_analyst_to_bull),
                                                attention_mask=attention_mask, position_ids=position_ids)
        h_bear = self.roles[Role.RESEARCH_BEAR](agg_from(self.edge_analyst_to_bear),
                                                attention_mask=attention_mask, position_ids=position_ids)

        # n 轮研究员“辩论” = 双向隐藏态互传 + 再过各自 node
        for _ in range(self.flow.debate.n_research_rounds - 1):
            h_b2b = self.edge_bull_to_bear(h_bull, attention_mask) if self.use_edges else h_bull
            h_be2b = self.edge_bear_to_bull(h_bear, attention_mask) if self.use_edges else h_bear
            h_bear = self.roles[Role.RESEARCH_BEAR](h_bear + h_b2b, attention_mask=attention_mask, position_ids=position_ids)
            h_bull = self.roles[Role.RESEARCH_BULL](h_bull + h_be2b, attention_mask=attention_mask, position_ids=position_ids)

        # 主持裁决（向量门控）：对各自平均池化后打分 softmax
        def pool_last(h: torch.Tensor):  # [B,T,C] -> [B,C]（末 token）
            return h[:, -1, :]

        gate_logits = self.research_gate(pool_last(h_bull) + pool_last(h_bear))  # 简单共享门控
        w = torch.softmax(gate_logits, dim=-1)  # [B,2]
        h_research = (w[:, 0].unsqueeze(1).unsqueeze(2) * h_bull) + (w[:, 1].unsqueeze(1).unsqueeze(2) * h_bear)

        # Trader：来自 Analysts 与 Research
        h_tr_in = []
        for r in self.flow.analyst_roles():
            h_tr_in.append(self.edge_analyst_to_trader[r.value](h_analyst_sum, attention_mask) if self.use_edges else h_analyst_sum)
        h_tr_in.append(self.edge_bull_to_trader(h_bull, attention_mask) if self.use_edges else h_bull)
        h_tr_in.append(self.edge_bear_to_trader(h_bear, attention_mask) if self.use_edges else h_bear)
        h_trader = self.roles[Role.TRADER](torch.stack(h_tr_in, 0).sum(0),
                                           attention_mask=attention_mask, position_ids=position_ids)

        # Risk 三节点：由 Trader 初始化
        h_risk: Dict[Role, torch.Tensor] = {}
        for r in self.flow.risk_roles():
            hin = self.edge_trader_to_risk[r.value](h_trader, attention_mask) if self.use_edges else h_trader
            h_risk[r] = self.roles[r](hin, attention_mask=attention_mask, position_ids=position_ids)

        # n 轮风控“讨论”：三方全连接互传
        for _ in range(self.flow.debate.n_risk_rounds - 1):
            new_states = {}
            for r_tgt in self.flow.risk_roles():
                acc = h_risk[r_tgt]
                for r_src in self.flow.risk_roles():
                    if r_src == r_tgt: continue
                    edge = self.edge_risk_pair[f"{r_src.value}->{r_tgt.value}"]
                    acc = acc + (edge(h_risk[r_src], attention_mask) if self.use_edges else h_risk[r_src])
                new_states[r_tgt] = self.roles[r_tgt](acc, attention_mask=attention_mask, position_ids=position_ids)
            h_risk = new_states

        # 风控主持门控（三类 softmax）
        risk_pooled = sum([pool_last(h) for h in h_risk.values()])  # 简单共享门控
        rlogits = self.risk_gate(risk_pooled)  # [B,3]
        rw = torch.softmax(rlogits, dim=-1)
        roles = self.flow.risk_roles()
        h_risk_mix = (
            rw[:, 0].unsqueeze(1).unsqueeze(2) * h_risk[roles[0]] +
            rw[:, 1].unsqueeze(1).unsqueeze(2) * h_risk[roles[1]] +
            rw[:, 2].unsqueeze(1).unsqueeze(2) * h_risk[roles[2]]
        )

        # Fund Manager：Trader + 风控合成
        h_fm_in = (self.edge_trader_to_fm(h_trader, attention_mask) if self.use_edges else h_trader) + \
                  (self.edge_riskmix_to_fm(h_risk_mix, attention_mask) if self.use_edges else h_risk_mix)
        h_fm = self.roles[Role.FUND_MANAGER](h_fm_in, attention_mask=attention_mask, position_ids=position_ids)

        # 归一化后接 LMHead
        h_dec = self.decision_norm(h_fm)
        logits = self.dst(h_dec)

        out = {"logits": logits}
        if return_hidden:
            out.update({
                "hidden_src": h_src,
                "hidden_analysts": {k.value: v for k, v in h_analysts.items()},
                "hidden_research_bull": h_bull,
                "hidden_research_bear": h_bear,
                "hidden_trader": h_trader,
                "hidden_risk": {k.value: v for k, v in h_risk.items()},
                "hidden_fund_manager": h_fm,
            })
        return out

    # 与 trainer/evaluate_text 兼容的接口
    def compute_lm_loss(
        self, input_ids, labels, attention_mask=None, position_ids=None, ignore_index=None
    ):
        if ignore_index is None:
            ignore_index = self.tokenizer.pad_token_id
        out = self.forward(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = out["logits"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss, logits

    @torch.no_grad()
    def greedy_decode(self, input_ids, max_new_tokens=64, attention_mask=None):
        self.eval()
        dev = input_ids.device
        for _ in range(max_new_tokens):
            out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            nxt = out["logits"][:, -1, :].argmax(-1, keepdim=True)
            input_ids = torch.cat([input_ids, nxt.to(dev)], dim=1)
            if attention_mask is not None:
                pad = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=dev)
                attention_mask = torch.cat([attention_mask, pad], dim=1)
        return input_ids

