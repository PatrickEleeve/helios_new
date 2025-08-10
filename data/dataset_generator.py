# data/dataset_generator.py
from __future__ import annotations
import random
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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

def synth_sample(label: Optional[str] = None,
                 p_buy: float = 0.34,
                 p_sell: float = 0.33,
                 p_hold: float = 0.33) -> Tuple[str, str]:
    """Create a single synthetic sample.

    Args:
        label: Optional fixed label among {BUY, SELL, HOLD}. If None, sample by probabilities.
        p_buy/p_sell/p_hold: Probabilities used when label is None. They will be normalized.

    Returns:
        (text, label)
    """
    ctx = random.choice(CONTEXTS)

    # Choose label deterministically if provided; else by probabilities
    if label is None:
        total = max(p_buy + p_sell + p_hold, 1e-9)
        probs = [p_buy / total, p_sell / total, p_hold / total]
        r = random.random()
        if r < probs[0]:
            label = "BUY"
        elif r < probs[0] + probs[1]:
            label = "SELL"
        else:
            label = "HOLD"

    if label == "BUY":
        sig = random.choice(SIGS_BULL)
        tpl = random.choice(BUY_TEMPLATES)
    elif label == "SELL":
        sig = random.choice(SIGS_BEAR)
        tpl = random.choice(SELL_TEMPLATES)
    else:
        sig = random.choice(SIGS_NEUTRAL)
        tpl = random.choice(HOLD_TEMPLATES)

    # Vary roles header: pick a random subset and shuffle order
    roles_pick = random.sample(ROLE_BUCKETS, k=random.randint(2, len(ROLE_BUCKETS)))
    random.shuffle(roles_pick)
    header = f"[TradingAgents Roles: {', '.join(roles_pick)}]"

    text = header + "\n" + tpl.format(ctx=ctx, sig=sig) + f"\nRationale: ...\nDecision: {label}\n"
    return text, label

def _row(text: str, label: str) -> Dict:
    uid = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return {"id": uid, "text": text, "label": label}

def gen_dataset(n_train: int,
                n_eval: int,
                seed: int = 42,
                balanced: bool = False,
                p_buy: float = 0.34,
                p_sell: float = 0.33,
                p_hold: float = 0.33) -> Tuple[List[Dict], List[Dict]]:
    """Generate synthetic dataset.

    If balanced=True, enforce near-equal class counts for BUY/SELL/HOLD.
    Otherwise, sample by probabilities.
    """
    random.seed(seed)
    train, evals = [], []

    if balanced:
        labels = ["BUY", "SELL", "HOLD"]
        # Train
        base, rem = divmod(n_train, 3)
        counts = {lbl: base for lbl in labels}
        for i in range(rem):
            counts[labels[i]] += 1
        for lbl in labels:
            for _ in range(counts[lbl]):
                t, y = synth_sample(label=lbl)
                train.append(_row(t, y))
        random.shuffle(train)
        # Eval
        base_e, rem_e = divmod(n_eval, 3)
        counts_e = {lbl: base_e for lbl in labels}
        for i in range(rem_e):
            counts_e[labels[i]] += 1
        for lbl in labels:
            for _ in range(counts_e[lbl]):
                t, y = synth_sample(label=lbl)
                evals.append(_row(t, y))
        random.shuffle(evals)
    else:
        for _ in range(n_train):
            t, y = synth_sample(p_buy=p_buy, p_sell=p_sell, p_hold=p_hold)
            train.append(_row(t, y))
        for _ in range(n_eval):
            t, y = synth_sample(p_buy=p_buy, p_sell=p_sell, p_hold=p_hold)
            evals.append(_row(t, y))
    return train, evals

def save_jsonl(path: str, rows: List[Dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Synthetic trading scenario dataset generator")
    parser.add_argument("--out-train", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "processed" / "trading_scenarios_train.jsonl"))
    parser.add_argument("--out-eval",  type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "processed" / "trading_scenarios_eval.jsonl"))
    parser.add_argument("--n-train",   type=int, default=2000)
    parser.add_argument("--n-eval",    type=int, default=300)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--balanced",  action="store_true", help="Enforce near-equal class balance")
    parser.add_argument("--p-buy",     type=float, default=0.34, help="BUY probability when not balanced")
    parser.add_argument("--p-sell",    type=float, default=0.33, help="SELL probability when not balanced")
    parser.add_argument("--p-hold",    type=float, default=0.33, help="HOLD probability when not balanced")
    args = parser.parse_args()

    train, evals = gen_dataset(
        n_train=args.n_train,
        n_eval=args.n_eval,
        seed=args.seed,
        balanced=args.balanced,
        p_buy=args.p_buy,
        p_sell=args.p_sell,
        p_hold=args.p_hold,
    )
    save_jsonl(args.out_train, train)
    save_jsonl(args.out_eval,  evals)

    # quick summary
    def dist(rows: List[Dict]) -> Dict[str, int]:
        out: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for r in rows:
            out[r["label"]] += 1
        return out

    print(f"[ok] wrote train={len(train)} {dist(train)} -> {args.out_train}")
    print(f"[ok] wrote eval={len(evals)} {dist(evals)} -> {args.out_eval}")

if __name__ == "__main__":
    main()
