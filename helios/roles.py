# helios/roles.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any

from .agents import AgentNode
from .protocols import DenseMode

class RoleModule(nn.Module):
    """
    角色节点包装：
    - 内含一个 AgentNode（仅 transformer core）
    - 预留 gate（标量门控）做极轻量的幅度调节；默认=1
    - 不做文本通信，pure hidden-state pipeline
    """
    def __init__(self, agent_node: AgentNode, hidden_size: int):
        super().__init__()
        self.node = agent_node
        self.hidden_size = hidden_size
        self.gate = nn.Parameter(torch.ones(1))  # 标量门控，默认 1

    def forward(
        self,
        incoming: List[torch.Tensor] | torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        dense_mode: DenseMode = DenseMode.DENSE,
        store_view: Optional[Dict[str, Any]] = None,  # 预留，不在 dense 模式使用
    ) -> torch.Tensor:
        h = self.node(incoming, attention_mask=attention_mask, position_ids=position_ids)
        return h * self.gate

