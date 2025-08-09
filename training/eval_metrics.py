# training/eval_metrics.py
from __future__ import annotations
import re, json
from typing import List, Tuple

BUY = {"buy","long","bullish"}
SELL= {"sell","short","bearish"}
HOLD= {"hold","neutral"}

def parse_decision(txt: str) -> str:
    s = txt.strip().lower()
    # 简单抓取末尾或关键词（根据你训练时的指令格式）
    if "decision:" in s:
        tail = s.split("decision:")[-1]
        s = tail.strip()
    for k in BUY:
        if k in s: return "BUY"
    for k in SELL:
        if k in s: return "SELL"
    for k in HOLD:
        if k in s: return "HOLD"
    # 兜底
    m = re.search(r"\b(buy|sell|hold)\b", s)
    return m.group(1).upper() if m else "HOLD"

def acc_sentiment(preds: List[str], golds: List[str]) -> float:
    ok = 0
    for p,g in zip(preds, golds):
        ok += int(parse_decision(p) == g)
    return ok / max(1, len(golds))

def exact_match_numeric(preds: List[str], golds: List[str]) -> float:
    def norm(x: str) -> str:
        x = x.strip()
        x = re.sub(r"[,$% ]","", x)
        return x.lower()
    ok = 0
    for p,g in zip(preds, golds):
        ok += int(norm(p) == norm(g))
    return ok / max(1, len(golds))
