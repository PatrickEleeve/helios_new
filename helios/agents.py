# helios/agents.py
import torch
import torch.nn as nn
from typing import List, Optional

class HFCoreWrapper(nn.Module):
    """
    将 HF 的 AutoModelForCausalLM 拆成三段：
    - embed(input_ids) -> hidden_states
    - core_forward(hidden_states, attention_mask, position_ids) -> hidden_states
    - logits(hidden_states) -> vocab logits
    适配 Qwen/LLaMA 系（model.model 为 transformer，model.model.embed_tokens 为嵌入，lm_head 为输出头）。
    """
    def __init__(self, causal_lm, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.inner = causal_lm
        self.transformer = getattr(causal_lm, "model", None) or getattr(causal_lm, "transformer")
        if self.transformer is None:
            raise RuntimeError("Unsupported model: cannot find .model or .transformer")
        self.embed_tokens = getattr(self.transformer, "embed_tokens", None)
        if self.embed_tokens is None:
            raise RuntimeError("Unsupported model: cannot find embed_tokens in transformer")
        self.lm_head = getattr(causal_lm, "lm_head", None)
        if self.lm_head is None:
            raise RuntimeError("Unsupported model: cannot find lm_head on CausalLM")
        self.hidden_size = int(self.transformer.config.hidden_size)

        if use_gradient_checkpointing and hasattr(self.transformer, "gradient_checkpointing_enable"):
            self.transformer.gradient_checkpointing_enable()

    @torch.no_grad()
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def core_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 通过 inputs_embeds 走“仅核心层”；不传 input_ids 即可跳过 embedding
        out = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
        )
        return out.last_hidden_state

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


class EmbeddingNode(nn.Module):
    """
    首节点：只做 embedding（保持与底模位置编码一致），附加LN稳定分布。
    """
    def __init__(self, core_wrapper: HFCoreWrapper):
        super().__init__()
        self.core = core_wrapper
        self.out_norm = nn.LayerNorm(self.core.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.core.embed(input_ids)  # [B,T,C]
        # 懒移动 LN 到输入的设备/精度（防止 CPU/CUDA 混用）
        if self.out_norm.weight.device != x.device or self.out_norm.weight.dtype != x.dtype:
            self.out_norm = self.out_norm.to(device=x.device, dtype=x.dtype)
        return self.out_norm(x)


class AgentNode(nn.Module):
    """
    中间节点：接收多入边隐藏态 -> sum 聚合 -> LN -> transformer core -> residual add
    """
    def __init__(self, core_wrapper: HFCoreWrapper):
        super().__init__()
        self.core = core_wrapper
        self.hidden_size = self.core.hidden_size
        self.input_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        incoming_states: List[torch.Tensor] | torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(incoming_states, torch.Tensor):
            agg = incoming_states
        else:
            assert len(incoming_states) > 0, "AgentNode: incoming_states is empty."
            agg = incoming_states[0] if len(incoming_states) == 1 else torch.stack(incoming_states, dim=0).sum(dim=0)

        # 懒移动 LN
        if self.input_norm.weight.device != agg.device or self.input_norm.weight.dtype != agg.dtype:
            self.input_norm = self.input_norm.to(device=agg.device, dtype=agg.dtype)

        residual = agg
        x = self.input_norm(agg)
        x = self.core.core_forward(x, attention_mask=attention_mask, position_ids=position_ids)
        return x + residual


class LMHeadNode(nn.Module):
    """
    末节点：把隐藏态投到词表 logits。
    """
    def __init__(self, core_wrapper: HFCoreWrapper):
        super().__init__()
        self.core = core_wrapper

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.core.logits(hidden_states)  # [B,T,V]
