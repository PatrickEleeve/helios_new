# helios/state_store.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

class EvidenceStore:
    """
    极简结构化状态库：你可替换为本地DB/Redis/向量库。
    - add(): 保存分角色的证据条目（任何结构化字段）
    - view_for_role(): 生成该角色可见的“摘要视图”（训练时只作为额外提示文本或轻量偏置）
    """
    def __init__(self):
        self._rows: List[Dict[str, Any]] = []

    def add(self, role: str, fields: Dict[str, Any]):
        row = dict(fields)
        row["_role"] = role
        self._rows.append(row)

    def query(self, role: Optional[str] = None, **filters) -> List[Dict[str, Any]]:
        res = []
        for r in self._rows:
            if role is not None and r.get("_role") != role:
                continue
            ok = True
            for k, v in filters.items():
                if r.get(k) != v:
                    ok = False; break
            if ok:
                res.append(r)
        return res

    def summarize_for_role(self, role: str, keys: Optional[list[str]] = None, max_items: int = 8) -> Dict[str, Any]:
        """
        生成角色可见的简要结构化视图；训练时可把它转成一行文本提示或用于轻量控制（如 gating/bias）。
        """
        items = self.query(role=role)[:max_items]
        if keys:
            items = [{k: it.get(k) for k in keys if k in it} for it in items]
        return {
            "role": role,
            "count": len(items),
            "items": items,
        }
