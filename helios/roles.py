# helios/roles.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any

from .agents import AgentNode
from .protocols import RoleConfig, DenseMode
from .state_store import EvidenceStore

class RoleModule(nn.Module):
    """
    角色节点包装：
    - 内含一个 AgentNode（仅 transformer core）
    - 可注入结构化状态视图（hybrid 模式）做轻量 gating/bias
    - 统一 forward 接口：incoming hidden states -> node -> role-biased hidden
    """
    def __init__(self, agent_node: AgentNode, cfg: RoleConfig, hidden_size: int):
        super().__init__()
        self.node = agent_node
        self.cfg = cfg
        self.hidden_size = hidden_size
        # 轻量可学习门控：把结构化视图编码成一组 gate（线性层），影响本角色输出幅度
        self.summary_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.ones(1))  # 标量门控，默认1

    def forward(
        self,
        incoming: List[torch.Tensor] | torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        dense_mode: DenseMode = DenseMode.DENSE,
        store_view: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        h = self.node(incoming, attention_mask=attention_mask, position_ids=position_ids)
        if dense_mode != DenseMode.DENSE and store_view is not None:
            # 简单把 view 的哈希/长度映射成一个向量门控（这里做超轻实现：用 count 做缩放）
            count = float(store_view.get("count", 0))
            scale = torch.clamp(torch.tensor(1.0 + 0.01 * count, device=h.device, dtype=h.dtype), 0.9, 1.2)
            h = h * (self.gate * scale)
        else:
            h = h * self.gate
        return h
