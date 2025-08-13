"""
FinQA to Helios format converter.

Reads a FinQA-like JSON/JSONL and writes data/processed/finqa_*.jsonl
Each output row has at least: {"text": "..."} where text starts with [SNAPSHOT]
so HeliosArchitecture's analyst masks can fall back to the snapshot segment.

Supported input shapes (best-effort):
- JSONL: one object per line with keys: question, answer, and optional context/pre_text/post_text.
- JSON: a list of the above objects.

Usage examples:
  python data/finqa_prepare.py --in path/to/finqa_train.jsonl --out data/processed/finqa_train.jsonl
  python data/finqa_prepare.py --in path/to/finqa_dev.jsonl   --out data/processed/finqa_eval.jsonl
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List


def _read_any(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    elif p.suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            # common variants: {"data": [...]}
            for k in ("data", "examples", "items"):  # heuristic
                if k in obj and isinstance(obj[k], list):
                    return obj[k]  # type: ignore
            # single object
            return [obj]
        raise ValueError("Unrecognized JSON structure for FinQA input")
    else:
        raise ValueError(f"Unsupported extension: {p.suffix}")


def _pick(d: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in d and isinstance(d[k], str) and d[k].strip():
            return d[k]
    return default


def _build_text(sample: Dict[str, Any]) -> str:
    # Extract fields with fallbacks
    q = _pick(sample, ["question", "query", "Problem", "Q"], default="")
    a = _pick(sample, ["answer", "final_answer", "gold", "A", "solution"], default="")
    ctx_parts: List[str] = []
    for key in ("context", "passage", "pre_text", "post_text", "evidence", "doc_text"):
        v = sample.get(key)
        if isinstance(v, str) and v.strip():
            ctx_parts.append(v.strip())
    # Some datasets include list of texts
    for key in ("contexts", "evidences"):
        v = sample.get(key)
        if isinstance(v, list):
            ctx_parts.extend([str(x) for x in v if isinstance(x, (str, int, float))])
    ctx = "\n".join(ctx_parts).strip()

    # Minimal prompt: ensure SNAPSHOT tag for valid masks
    parts: List[str] = ["[SNAPSHOT]"]
    if ctx:
        parts.append(f"Context: {ctx}")
    if q:
        parts.append(f"Question: {q}")
    # Train as LM: include the answer in the text so NTP can learn to generate it
    if a:
        parts.append(f"Answer: {a}")
    return "\n".join(parts) + "\n"


def convert(in_path: str, out_path: str) -> int:
    rows = _read_any(in_path)
    out_rows: List[Dict[str, str]] = []
    for r in rows:
        try:
            txt = _build_text(r)
            out_rows.append({"text": txt})
        except Exception:
            # best effort; skip malformed rows
            continue
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(out_rows)


def main():
    parser = argparse.ArgumentParser(description="Convert FinQA-like data to Helios JSONL format")
    parser.add_argument("--in", dest="in_path", required=True, help="Input FinQA json/jsonl file")
    parser.add_argument("--out", dest="out_path", default=str(Path(__file__).resolve().parents[1] / "data" / "processed" / "finqa_train.jsonl"))
    args = parser.parse_args()
    n = convert(args.in_path, args.out_path)
    print(f"[ok] wrote {n} rows -> {args.out_path}")

if __name__ == "__main__":
    main()
