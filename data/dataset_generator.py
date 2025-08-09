# data/dataset_generator.py
from __future__ import annotations
import random, json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

ROLE_BUCKETS = ["research", "fundamental", "quant", "risk"]

BUY_TEMPLATES = [
    "Context: {ctx}\nSignal: {sig}\nTask: Produce a concise investment plan ending with 'Decision: BUY'.",
    "Market brief: {ctx}\nRationale: {sig}\nRespond with a plan. Decision: BUY.",
]
SELL_TEMPLATES = [
    "Context: {ctx}\nAlert: {sig}\nTask: Provide risk rationale and end with 'Decision: SELL'.",
    "Market brief: {ctx}\nBearish signal: {sig}\nOutput action. Decision: SELL.",
]
HOLD_TEMPLATES = [
    "Context: {ctx}\nMixed signal: {sig}\nTask: Provide reasoning and end with 'Decision: HOLD'.",
    "Market brief: {ctx}\nUnclear: {sig}\nRecommendation: Decision: HOLD.",
]

CONTEXTS = [
    "Earnings approaching with implied vol elevated; options skew widened.",
    "Macro CPI surprise; 10Y yields spiking; USD strengthening.",
    "Sector rotation into defensives; breadth deteriorating.",
    "Rumors of regulatory probe; management change imminent.",
    "AI capex cycle accelerating; supply chain easing.",
]

SIGS_BULL = [
    "EPS revisions trending up; new orders accelerating.",
    "Golden cross on daily; strong momentum with volume confirmation.",
    "Buyback authorization increased; insider purchasing observed.",
]
SIGS_BEAR = [
    "Revenue miss pre-announced; gross margin compression seen.",
    "Breakdown below 200DMA; abnormal put/call ratio.",
    "Credit spread widening; downgrade risk high.",
]
SIGS_NEUTRAL = [
    "Mixed guidance; valuation near peers; positioning neutral.",
    "Mean-reverting volatility; RSI mid-range; no catalyst.",
    "Flows balanced; sentiment surveys neutral.",
]

def synth_sample() -> Tuple[str, str]:
    ctx = random.choice(CONTEXTS)
    choice = random.random()
    if choice < 0.34:
        sig = random.choice(SIGS_BULL)
        tpl = random.choice(BUY_TEMPLATES)
        label = "BUY"
    elif choice < 0.67:
        sig = random.choice(SIGS_BEAR)
        tpl = random.choice(SELL_TEMPLATES)
        label = "SELL"
    else:
        sig = random.choice(SIGS_NEUTRAL)
        tpl = random.choice(HOLD_TEMPLATES)
        label = "HOLD"

    # 伪“角色协作”提示（纯文本，用于自回归）
    roles = ", ".join(ROLE_BUCKETS)
    header = f"[TradingAgents Roles: {roles}]"
    text = header + "\n" + tpl.format(ctx=ctx, sig=sig) + f"\nRationale: ...\nDecision: {label}\n"
    return text, label

def gen_dataset(n_train: int, n_eval: int, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    random.seed(seed)
    train, evals = [], []
    for _ in range(n_train):
        t, y = synth_sample()
        train.append({"text": t, "label": y})
    for _ in range(n_eval):
        t, y = synth_sample()
        evals.append({"text": t, "label": y})
    return train, evals

def save_jsonl(path: str, rows: List[Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-train", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "processed" / "trading_scenarios_train.jsonl"))
    parser.add_argument("--out-eval",  type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "processed" / "trading_scenarios_eval.jsonl"))
    parser.add_argument("--n-train",   type=int, default=2000)
    parser.add_argument("--n-eval",    type=int, default=300)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    train, evals = gen_dataset(args.n_train, args.n_eval, args.seed)
    save_jsonl(args.out-train, train)   # type: ignore
    save_jsonl(args.out-eval,  evals)   # type: ignore
    print(f"[ok] wrote train={len(train)} eval={len(evals)}")

if __name__ == "__main__":
    main()