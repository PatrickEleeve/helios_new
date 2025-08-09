# evaluate_text.py
from __future__ import annotations
import argparse, json, os, glob
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from helios.architecture import HeliosArchitecture

def _resolve_ckpt_path(p: str | None) -> str | None:
    if not p:
        return None
    if os.path.isdir(p):
        # 允许传 outputs/exp_xxx 或 outputs/exp_xxx/step_xxx
        cand = sorted(glob.glob(os.path.join(p, "step_*", "ckpt.pt")))
        if cand:
            return cand[-1]
        pt = os.path.join(p, "ckpt.pt")
        if os.path.exists(pt):
            return pt
        # 传的是 step_yyy 目录
        cand2 = os.path.join(p, "ckpt.pt")
        if os.path.exists(cand2):
            return cand2
        raise FileNotFoundError(f"No ckpt.pt under: {p}")
    else:
        return p  # 直接是文件

def load_arch(model_id: str, middle_nodes: int, ckpt: Optional[str], dtype: str = "bf16", use_edges: bool = True):
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    arch = HeliosArchitecture(
        model_id=model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        middle_nodes=middle_nodes,
        use_edges=use_edges,
    )
    ckpt_path = _resolve_ckpt_path(ckpt) if ckpt else None
    if ckpt_path:
        print(f"[load] {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        arch.load_state_dict(state["state_dict"], strict=False)
    arch.eval()
    return arch

def split_prompt(text: str, prefer_prefix: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
    """
    返回: (prompt_without_answer, gold_answer_if_any, mode)
    mode in {"decision","answer",None}
    """
    t = text
    if prefer_prefix is not None and prefer_prefix in t:
        pre, post = t.split(prefer_prefix, 1)
        return (pre + prefer_prefix, post.strip(), "decision" if "Decision" in prefer_prefix else "answer")

    if "Decision:" in t:
        pre, post = t.split("Decision:", 1)
        return (pre + "Decision:", post.strip(), "decision")
    if "Answer:" in t:
        pre, post = t.split("Answer:", 1)
        return (pre + "Answer:", post.strip(), "answer")
    return (t, None, None)

def decode_batch(arch: HeliosArchitecture, prompts: List[str], max_new: int = 64) -> List[str]:
    tok = arch.tokenizer
    res = []
    for p in prompts:
        enc = tok(p, return_tensors="pt")
        input_ids = enc["input_ids"].to("cuda")
        attn = enc["attention_mask"].to("cuda")
        out_ids = arch.greedy_decode(input_ids, max_new_tokens=max_new, attention_mask=attn)
        txt = tok.decode(out_ids[0], skip_special_tokens=True)
        res.append(txt)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-8B")
    ap.add_argument("--middle-nodes", type=int, default=2)
    ap.add_argument("--ckpt", type=str, default=None, help="outputs/.../ 或 outputs/.../step_xxx/ 或 ckpt.pt")
    ap.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    ap.add_argument("--no-edges", action="store_true", help="推理时绕过 EdgeAdapters（等价底模前向）")

    # 单条 or 批量
    ap.add_argument("--prompt", type=str, help="单条测试：直接给到 Decision:/Answer: 结尾")
    ap.add_argument("--file", type=str, help="批量测试：jsonl，每行包含 {\"text\": ...}")
    ap.add_argument("--field", type=str, default="text")
    ap.add_argument("--prefix", type=str, default=None, help="强制用哪个前缀切金标准，例如 'Decision:' 或 'Answer:'")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--out", type=str, default="preds.jsonl")

    args = ap.parse_args()

    arch = load_arch(args.model_id, args.middle_nodes, args.ckpt, args.dtype, use_edges=not args.no_edges)
    tok = arch.tokenizer

    # 单条
    if args.prompt:
        prompt, _, _ = split_prompt(args.prompt, args.prefix)
        enc = tok(prompt, return_tensors="pt")
        out_ids = arch.greedy_decode(enc["input_ids"].to("cuda"), max_new_tokens=args.max_new,
                                     attention_mask=enc["attention_mask"].to("cuda"))
        print(tok.decode(out_ids[0], skip_special_tokens=True))
        return

    # 批量
    assert args.file, "请提供 --prompt 或 --file"
    preds, golds, raw_prompts, full_texts = [], [], [], []

    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            text = obj.get(args.field, obj.get("text", ""))
            prompt, gold, mode = split_prompt(text, args.prefix)
            raw_prompts.append((prompt, mode))
            golds.append(gold)
            full_texts.append(text)

    decoded = decode_batch(arch, [p for p,_ in raw_prompts], max_new=args.max_new)

    for dec, (prompt, mode) in zip(decoded, raw_prompts):
        tail = dec.split(prompt, 1)[1].strip() if prompt in dec else dec
        preds.append(tail)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        for text, prompt, pred, gold in zip(full_texts, [p for p,_ in raw_prompts], preds, golds):
            w.write(json.dumps({"text": text, "prompt": prompt, "prediction": pred, "gold": gold}, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {args.out}")

if __name__ == "__main__":
    main()

