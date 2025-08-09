# helios/protocols.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List

class Role(str, Enum):
    # Analysts (4)
    ANALYST_FUND = "analyst_fundamental"
    ANALYST_SENT = "analyst_sentiment"
    ANALYST_NEWS = "analyst_news"
    ANALYST_TECH = "analyst_technical"
    # Researchers (2)
    RESEARCH_BULL = "research_bull"
    RESEARCH_BEAR = "research_bear"
    # Trader
    TRADER = "trader"
    # Risk (3)
    RISK_RISKY   = "risk_risky"
    RISK_NEUTRAL = "risk_neutral"
    RISK_SAFE    = "risk_safe"
    # Fund manager
    FUND_MANAGER = "fund_manager"

class DenseMode(str, Enum):
    DENSE  = "dense"   # 仅隐藏态通信
    HYBRID = "hybrid"  # 可选：注入结构化视图（不产生文本）

@dataclass
class RoleConfig:
    name: Role
    hidden_bias_scale: float = 1.0
    enabled: bool = True

@dataclass
class DebateConfig:
    n_research_rounds: int = 3
    n_risk_rounds: int = 3

@dataclass
class FlowConfig:
    dense_mode: DenseMode = DenseMode.DENSE
    use_edges: bool = True
    debate: DebateConfig = DebateConfig()

    # 固定为 TradingAgents 的边拓扑（严格一致）
    def analyst_roles(self) -> List[Role]:
        return [Role.ANALYST_FUND, Role.ANALYST_SENT, Role.ANALYST_NEWS, Role.ANALYST_TECH]

    def researcher_roles(self) -> List[Role]:
        return [Role.RESEARCH_BULL, Role.RESEARCH_BEAR]

    def risk_roles(self) -> List[Role]:
        return [Role.RISK_RISKY, Role.RISK_NEUTRAL, Role.RISK_SAFE]
