# helios/architecture.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from .adapters import EdgeAdapter
from .agents import HFCoreWrapper, EmbeddingNode, AgentNode, LMHeadNode
from .protocols import Role, FlowConfig, DenseMode

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"

SECTION_TAGS = {
    Role.ANALYST_FUND: "[FUND]",
    Role.ANALYST_SENT: "[SENT]",
    Role.ANALYST_NEWS: "[NEWS]",
    Role.ANALYST_TECH: "[TECH]",
    # 可选：通用头部
    "SNAPSHOT": "[SNAPSHOT]",
}

class HeliosArchitecture(nn.Module):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device_map: str | dict | None = "auto",
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        trust_remote_code: bool = True,
        # 为兼容旧调用
        middle_nodes: Optional[int] = None,
        use_edges: Optional[bool] = None,
        # 正式配置
        flow: Optional[FlowConfig] = None,
        gradient_checkpointing: bool = False,
        edge_nhead: int = 8,
        edge_ffn_mult: int = 4,
        edge_dropout: float = 0.0,
        # 新增：是否对 Analyst 启用“分区掩码”
        enforce_analyst_section_masks: bool = True,
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
        if use_edges is not None:
            self.flow.use_edges = bool(use_edges)
        self.use_edges = bool(self.flow.use_edges)
        self.dense_mode = self.flow.dense_mode
        self.enforce_masks = enforce_analyst_section_masks

        # ---- 节点 ----
        self.src = EmbeddingNode(self.core)

        def _mk_role() -> AgentNode:
            return AgentNode(self.core)

        # Analysts 4
        self.roles: Dict[Role, AgentNode] = {}
        for r in self.flow.analyst_roles():
            self.roles[r] = _mk_role()

        # Researchers 2
        for r in self.flow.researcher_roles():
            self.roles[r] = _mk_role()

        # Trader
        self.roles[Role.TRADER] = _mk_role()

        # Risk 3
        for r in self.flow.risk_roles():
            self.roles[r] = _mk_role()

        # Fund Manager
        self.roles[Role.FUND_MANAGER] = _mk_role()

        # 输出头
        self.decision_norm = nn.LayerNorm(self.h)
        self.dst = LMHeadNode(self.core)

        # ---- 边（EdgeAdapter）----
        def EA(): return EdgeAdapter(self.h, self.h, nhead=edge_nhead, ffn_mult=edge_ffn_mult, dropout=edge_dropout)

        # src -> Analyst（每个分析师各一条；但输入会先被“分区掩码”裁剪）
        self.edge_src_to_analyst = nn.ModuleDict({r.value: EA() for r in self.flow.analyst_roles()})

        # Analyst -> Researchers（逐分析师单独路由）
        self.edge_analyst_to_bull = nn.ModuleDict({r.value: EA() for r in self.flow.analyst_roles()})
        self.edge_analyst_to_bear = nn.ModuleDict({r.value: EA() for r in self.flow.analyst_roles()})

        # Researchers 双向
        self.edge_bull_to_bear = EA()
        self.edge_bear_to_bull = EA()

        # Researchers -> Trader
        self.edge_bull_to_trader = EA()
        self.edge_bear_to_trader = EA()
        # Analyst -> Trader（逐分析师）
        self.edge_analyst_to_trader = nn.ModuleDict({r.value: EA() for r in self.flow.analyst_roles()})

        # Trader -> Risk（每个风险节点一条）
        self.edge_trader_to_risk = nn.ModuleDict({r.value: EA() for r in self.flow.risk_roles()})

        # Risk 三方全连接（不含自环）
        self.edge_risk_pair = nn.ModuleDict()
        risk_list = self.flow.risk_roles()
        for i in range(len(risk_list)):
            for j in range(len(risk_list)):
                if i == j: continue
                self.edge_risk_pair[f"{risk_list[i].value}->{risk_list[j].value}"] = EA()

        # Trader & RiskMix -> FundManager
        self.edge_trader_to_fm = EA()
        self.edge_riskmix_to_fm = EA()

        # 主持门控：研究员 2 类 + 风控 3 类
        self.research_gate = nn.Linear(self.h, 2, bias=False)
        self.risk_gate = nn.Linear(self.h, 3, bias=False)

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

    # ---------- 实用：按标签切分区间并构造掩码 ----------
    def _find_section_spans(
        self, input_ids: torch.Tensor
    ) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        返回每个标签的起止位置（闭区间）列表（按 batch 展开）。
        约定：标签格式为独立 token（例如 [FUND]），段落覆盖从该标签到下一个标签/结尾为止。
        """
        B, T = input_ids.shape
        tags = {k: self.tokenizer.encode(v, add_special_tokens=False)[0] for k, v in SECTION_TAGS.items()}
        spans = {k: [] for k in tags.keys()}

        # 先找到每个位置是否为任一标签
        tag_pos = {k: (input_ids == tid) for k, tid in tags.items()}  # [B,T] bool

        # 对每个 batch，按出现顺序切段
        for b in range(B):
            positions = []
            for name, mask in tag_pos.items():
                idxs = torch.nonzero(mask[b], as_tuple=False).flatten()
                for i in idxs:
                    positions.append((int(i.item()), name))
            if not positions:
                continue
            positions.sort(key=lambda x: x[0])
            for i, (start, name) in enumerate(positions):
                end = (positions[i+1][0] - 1) if i+1 < len(positions) else (T - 1)
                spans[name].append((torch.tensor(b), torch.tensor(start), torch.tensor(end)))
        return spans

    def _build_analyst_masks(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> Dict[Role, torch.Tensor]:
        """
        为每位 Analyst 生成 [B,T,1] 掩码：属于该段=1，其它=0；若该段缺失，退化为使用 [SNAPSHOT] 段。
        """
        B, T = input_ids.shape
        spans = self._find_section_spans(input_ids)
        masks: Dict[Role, torch.Tensor] = {}

        def make_mask(tag_name: str) -> torch.Tensor:
            m = torch.zeros((B, T), dtype=torch.bool, device=input_ids.device)
            for (b, s, e) in spans.get(tag_name, []):
                b = int(b.item()); s = int(s.item()); e = int(e.item())
                m[b, s:e+1] = True
            return m

        snap_m = make_mask("SNAPSHOT")
        for role, tag in [
            (Role.ANALYST_FUND, "ANALYST_FUND"),
            (Role.ANALYST_SENT, "ANALYST_SENT"),
            (Role.ANALYST_NEWS, "ANALYST_NEWS"),
            (Role.ANALYST_TECH, "ANALYST_TECH"),
        ]:
            # 映射 tag key
            tag_key = role
            tag_name = tag_key  # Role → key
            # 取对应段；若为空则用 SNAPSHOT 段
            role_tag = SECTION_TAGS[role]
            m_role = make_mask(role_tag.strip("[]"))
            # 兼容：上面 make_mask 用 key 名，这里做一次 fallback
            if not m_role.any():
                m_role = snap_m
            # 应用 attention_mask（pad 位置强制0）
            if attention_mask is not None:
                m_role = m_role & (attention_mask.bool())
            masks[role] = m_role.unsqueeze(-1).to(dtype=torch.float32)  # [B,T,1]
        return masks

    # ---- 辅助：取最后一个非 PAD token 的向量 ----
    @staticmethod
    def _pool_last_nonpad(h: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # h: [B, T, C], attention_mask: [B, T] (1=real, 0=pad)
        if attention_mask is None:
            return h[:, -1, :]
        lengths = attention_mask.sum(dim=-1)  # [B]
        idx = (lengths - 1).clamp(min=0).to(torch.long)  # [B]
        B = h.size(0)
        return h[torch.arange(B, device=h.device), idx, :]  # [B, C]

    # --- 前向 ---
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:

        # 1) t=0 明文 → embedding
        h_src = self.src(input_ids, attention_mask=attention_mask, position_ids=position_ids)  # [B,T,C]

        # 2) Analyst 分区掩码：每位分析师只看自己的实时数据（段内 token）
        if self.enforce_masks:
            masks = self._build_analyst_masks(input_ids, attention_mask)  # [B,T,1] for each role
        else:
            masks = {r: torch.ones_like(h_src[..., :1]) for r in self.flow.analyst_roles()}

        # 3) Analysts：分区裁剪后的隐藏态 → 对应边 → 角色节点
        h_analysts: Dict[Role, torch.Tensor] = {}
        for r in self.flow.analyst_roles():
            # 只保留该段 token 的向量（其他位置为0）；确保掩码与隐藏态同精度，避免 dtype 提升
            h_masked = h_src * masks[r].to(dtype=h_src.dtype)
            hin = self.edge_src_to_analyst[r.value](h_masked, attention_mask) if self.use_edges else h_masked
            h_analysts[r] = self.roles[r](hin, attention_mask=attention_mask, position_ids=position_ids)

        # 4) Researchers 初始化：逐分析师路由到 Bull / Bear，然后在接收端求和
        def agg_from(md: nn.ModuleDict) -> torch.Tensor:
            outs = []
            for r in self.flow.analyst_roles():
                x = h_analysts[r]
                outs.append(md[r.value](x, attention_mask) if self.use_edges else x)
            return torch.stack(outs, dim=0).sum(dim=0)

        h_bull = self.roles[self.flow.researcher_roles()[0]](
            agg_from(self.edge_analyst_to_bull), attention_mask=attention_mask, position_ids=position_ids
        )
        h_bear = self.roles[self.flow.researcher_roles()[1]](
            agg_from(self.edge_analyst_to_bear), attention_mask=attention_mask, position_ids=position_ids
        )

        # 5) n 轮研究员互传
        for _ in range(self.flow.debate.n_research_rounds - 1):
            b2be = self.edge_bull_to_bear(h_bull, attention_mask) if self.use_edges else h_bull
            be2b = self.edge_bear_to_bull(h_bear, attention_mask) if self.use_edges else h_bear
            h_bear = self.roles[self.flow.researcher_roles()[1]](h_bear + b2be, attention_mask=attention_mask, position_ids=position_ids)
            h_bull = self.roles[self.flow.researcher_roles()[0]](h_bull + be2b, attention_mask=attention_mask, position_ids=position_ids)

        # 6) 研究主持门控（最后一个非 PAD token）
        rb = self._pool_last_nonpad(h_bull, attention_mask)
        rr = self._pool_last_nonpad(h_bear, attention_mask)
        gate_logits = self.research_gate(rb + rr)       # [B,2]
        w = torch.softmax(gate_logits, dim=-1)           # [B,2]
        h_research = (w[:, 0].unsqueeze(1).unsqueeze(2) * h_bull) + (w[:, 1].unsqueeze(1).unsqueeze(2) * h_bear)

        # 7) Trader：来自各 analyst（逐条边） + bull/bear
        h_tr_in = []
        for r in self.flow.analyst_roles():
            x = h_analysts[r]
            h_tr_in.append(self.edge_analyst_to_trader[r.value](x, attention_mask) if self.use_edges else x)
        h_tr_in.append(self.edge_bull_to_trader(h_bull, attention_mask) if self.use_edges else h_bull)
        h_tr_in.append(self.edge_bear_to_trader(h_bear, attention_mask) if self.use_edges else h_bear)
        h_trader = self.roles[Role.TRADER](torch.stack(h_tr_in, 0).sum(0),
                                           attention_mask=attention_mask, position_ids=position_ids)

        # 8) Risk 三节点初始化（由 Trader 态）
        h_risk: Dict[Role, torch.Tensor] = {}
        for r in self.flow.risk_roles():
            hin = self.edge_trader_to_risk[r.value](h_trader, attention_mask) if self.use_edges else h_trader
            h_risk[r] = self.roles[r](hin, attention_mask=attention_mask, position_ids=position_ids)

        # 9) n 轮风控互传
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

        # 10) 风控主持门控
        rp = sum([self._pool_last_nonpad(h, attention_mask) for h in h_risk.values()])
        rlogits = self.risk_gate(rp)        # [B,3]
        rw = torch.softmax(rlogits, dim=-1) # [B,3]
        roles = self.flow.risk_roles()
        h_risk_mix = (
            rw[:, 0].unsqueeze(1).unsqueeze(2) * h_risk[roles[0]] +
            rw[:, 1].unsqueeze(1).unsqueeze(2) * h_risk[roles[1]] +
            rw[:, 2].unsqueeze(1).unsqueeze(2) * h_risk[roles[2]]
        )

        # 11) Fund Manager：Trader + 风控合成
        h_fm_in = (self.edge_trader_to_fm(h_trader, attention_mask) if self.use_edges else h_trader) + \
                  (self.edge_riskmix_to_fm(h_risk_mix, attention_mask) if self.use_edges else h_risk_mix)
        h_fm = self.roles[Role.FUND_MANAGER](h_fm_in, attention_mask=attention_mask, position_ids=position_ids)

        # 12) 输出自然语言
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

    # 与 trainer/evaluate_text 兼容
    def compute_lm_loss(self, input_ids, labels, attention_mask=None, position_ids=None, ignore_index=None):
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

    # -----------------------------
    # 参数分组与冻结控制（供 Trainer 使用）
    # -----------------------------
    def edges_parameters(self):
        """
        返回所有“边”与门控相关的参数迭代器：EdgeAdapter 模块 + 研究/风控门控线性层。
        （这些是 Phase-1 仅训练的部分。）
        """
        modules = [
            self.edge_bull_to_bear, self.edge_bear_to_bull,
            self.edge_bull_to_trader, self.edge_bear_to_trader,
            self.edge_trader_to_fm, self.edge_riskmix_to_fm,
            self.research_gate, self.risk_gate,
        ]
        modules.extend(list(self.edge_src_to_analyst.values()))
        modules.extend(list(self.edge_analyst_to_bull.values()))
        modules.extend(list(self.edge_analyst_to_bear.values()))
        modules.extend(list(self.edge_analyst_to_trader.values()))
        modules.extend(list(self.edge_trader_to_risk.values()))
        modules.extend(list(self.edge_risk_pair.values()))

        for m in modules:
            for p in m.parameters():
                yield p

    def vertices_parameters(self):
        """
        返回图中“顶点”相关的参数：
        - 核心 Transformer（共享于所有 AgentNode）
        - 各 AgentNode/EmbeddingNode/LMHeadNode 上的小型层（例如 LayerNorm, lm_head）
        - 决策归一化层
        """
        # 核心 Transformer 与 lm_head
        for p in self.core.transformer.parameters():
            yield p
        for p in self.core.lm_head.parameters():
            yield p

        # 节点自身的小层
        for p in self.src.parameters():
            yield p
        for node in self.roles.values():
            for p in node.parameters():
                yield p
        for p in self.dst.parameters():
            yield p
        for p in self.decision_norm.parameters():
            yield p

    def freeze_vertices(self):
        """冻结顶点（核心与节点）的参数，仅训练边相关参数。"""
        for p in self.vertices_parameters():
            p.requires_grad = False
        for p in self.edges_parameters():
            p.requires_grad = True

    def unfreeze_vertices(self):
        """解冻顶点，允许同时训练边与顶点。"""
        for p in self.vertices_parameters():
            p.requires_grad = True
