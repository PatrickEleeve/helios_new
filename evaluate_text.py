# evaluate_text.py
from __future__ import annotations
import argparse, json, os, glob
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from helios.architecture import HeliosArchitecture
from helios.protocols import FlowConfig, DebateConfig, DenseMode

# -------- 从 configs 解析模型ID：优先 configs/base_config.py，再退回根目录 config.py --------
def _model_from_configs(cli_override: Optional[str]) -> str:
    if cli_override:  # 允许手动覆盖，但通常不需要
        return cli_override
    for mod in ("configs.base_config", "config"):
        try:
            m = __import__(mod, fromlist=["*"])
            for attr in ("MODELNAME", "DEFAULT_MODEL_ID", "MODEL_ID"):
                if hasattr(m, attr):
                    return getattr(m, attr)
        except Exception:
            continue
    # 兜底
    return "Qwen/Qwen3-8B"

def _resolve_ckpt_path(p: str | None) -> str | None:
    if not p:
        return None
    if os.path.isdir(p):
        cand = sorted(glob.glob(os.path.join(p, "step_*", "ckpt.pt")))
        if cand:
            return cand[-1]
        pt = os.path.join(p, "ckpt.pt")
        if os.path.exists(pt):
            return pt
        raise FileNotFoundError(f"No ckpt.pt under: {p}")
    else:
        return p

def load_arch(model_id: Optional[str], ckpt: Optional[str], dtype: str = "bf16",
              use_edges: bool = True, dense_mode: str = "dense",
              n_research_rounds: int = 3, n_risk_rounds: int = 3):
    model_id = _model_from_configs(model_id)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    flow = FlowConfig(
        dense_mode=DenseMode.DENSE if dense_mode == "dense" else DenseMode.HYBRID,
        use_edges=use_edges,
        debate=DebateConfig(n_research_rounds=n_research_rounds, n_risk_rounds=n_risk_rounds),
    )
    print(f"[model] using: {model_id}")
    arch = HeliosArchitecture(
        model_id=model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        flow=flow,
    )
    ckpt_path = _resolve_ckpt_path(ckpt) if ckpt else None
    if ckpt_path:
        print(f"[load] {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("arch") or state.get("state_dict") or state
        arch.load_state_dict(sd, strict=False)
    arch.eval()
    return arch

def split_prompt(text: str, prefer_prefix: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
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

def _pick_device() -> str:
    try:
        import torch as _t
        return "cuda" if _t.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def greedy_decode(arch: HeliosArchitecture, prompt: str, max_new: int = 64) -> str:
    tok = arch.tokenizer
    enc = tok(prompt, return_tensors="pt")
    dev = _pick_device()
    input_ids = enc["input_ids"].to(dev)
    attn = enc["attention_mask"].to(dev)
    out_ids = arch.greedy_decode(input_ids, max_new_tokens=max_new, attention_mask=attn)
    full = tok.decode(out_ids[0], skip_special_tokens=True)
    tail = full.split(prompt, 1)[1].strip() if prompt in full else full
    return tail

@torch.no_grad()
def rank_choices(arch: HeliosArchitecture, prompt: str, choices: List[str], score_norm: str = "none") -> str:
    """
    P(choice | prompt) 逐token打分，返回分数最高者。适合 BUY/HOLD/SELL。
    """
    tok = arch.tokenizer
    base = tok(prompt, return_tensors="pt")
    dev = _pick_device()
    base_ids = base["input_ids"].to(dev)
    base_attn = base["attention_mask"].to(dev)
    scores = []
    for ch in choices:
        ch_ids = tok(" " + ch, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(dev)
        ctx_ids = base_ids.clone(); ctx_attn = base_attn.clone()
        logp = 0.0
        for tid in ch_ids:
            out = arch.forward(input_ids=ctx_ids, attention_mask=ctx_attn)["logits"]  # [B,T,V]
            lp = F.log_softmax(out[:, -1, :], dim=-1)
            logp += lp[0, int(tid)].item()
            tid_ = tid.view(1,1)
            ctx_ids = torch.cat([ctx_ids, tid_.to(ctx_ids.device)], dim=1)
            one = torch.ones((ctx_attn.size(0),1), dtype=ctx_attn.dtype, device=ctx_attn.device)
            ctx_attn = torch.cat([ctx_attn, one], dim=1)
        if score_norm == "length" and len(ch_ids) > 0:
            logp = logp / len(ch_ids)
        scores.append(logp)
    best_idx = int(torch.tensor(scores).argmax().item())
    return choices[best_idx]

@torch.no_grad()
def constrained_first_token(arch: HeliosArchitecture, prompt: str, choices: List[str]) -> str:
    tok = arch.tokenizer
    enc = tok(prompt, return_tensors="pt")
    dev = _pick_device()
    input_ids = enc["input_ids"].to(dev)
    attn = enc["attention_mask"].to(dev)
    allowed = []
    for ch in choices:
        ids = tok(" " + ch, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if len(ids) > 0:
            allowed.append(int(ids[0]))
    allowed = sorted(set(allowed))
    out = arch.forward(input_ids=input_ids, attention_mask=attn)["logits"]
    last = out[:, -1, :]
    mask = torch.full_like(last, float("-inf")); mask[:, allowed] = last[:, allowed]
    next_id = mask.argmax(dim=-1, keepdim=True)
    new_ids = torch.cat([input_ids, next_id.to(input_ids.device)], dim=1)
    full = tok.decode(new_ids[0], skip_special_tokens=True)
    tail = full.split(tok.decode(input_ids[0], skip_special_tokens=True), 1)[1].strip()
    t = tail.strip().lower()
    for ch in choices:
        if ch.lower() in t:
            return ch
    return tail

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=None, help="通常留空：默认从 configs 读取（MODELNAME/DEFAULT_MODEL_ID）")
    ap.add_argument("--ckpt", type=str, default=None, help="outputs/.../ 或 outputs/.../step_xxx/ 或 ckpt.pt")
    ap.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    ap.add_argument("--no-edges", action="store_true", help="推理时绕过 EdgeAdapters（隐藏态直连）")
    ap.add_argument("--dense-mode", choices=["dense","hybrid"], default="dense")
    ap.add_argument("--research-rounds", type=int, default=3)
    ap.add_argument("--risk-rounds", type=int, default=3)
    ap.add_argument("--choice-mode", choices=["none","rank","constrained"], default="rank",
                    help="rank=对候选打分；constrained=受限首token；none=普通生成")
    ap.add_argument("--score-norm", choices=["none","length"], default="none", help="rank模式分数归一化方式")
    ap.add_argument("--choice-set", type=str, default="BUY,HOLD,SELL")
    ap.add_argument("--prompt", type=str, help="单条测试：建议在 Decision: 后留空格")
    ap.add_argument("--file", type=str, help="批量测试：jsonl，每行包含 {\"text\": ...}")
    ap.add_argument("--field", type=str, default="text")
    ap.add_argument("--prefix", type=str, default=None)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--out", type=str, default="preds.jsonl")  # ← 修正：type=str
    args = ap.parse_args()

    arch = load_arch(
        model_id=args.model_id,
        ckpt=args.ckpt,
        dtype=args.dtype,
        use_edges=not args.no_edges,
        dense_mode=args.dense_mode,
        n_research_rounds=args.research_rounds,
        n_risk_rounds=args.risk_rounds,
    )
    choices = [c.strip() for c in args.choice_set.split(",") if c.strip()]

    if args.prompt:
        prompt, _, _ = split_prompt(args.prompt, args.prefix)
        if args.choice_mode == "rank":
            print(rank_choices(arch, prompt, choices, score_norm=args.score_norm)); return
        if args.choice_mode == "constrained":
            print(constrained_first_token(arch, prompt, choices)); return
        print(greedy_decode(arch, prompt, max_new=args.max_new)); return

    assert args.file, "请提供 --prompt 或 --file"
    preds, golds, raw_prompts, full_texts = [], [], [], []
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            text = obj.get(args.field, obj.get("text", ""))
            prompt, gold, mode = split_prompt(text, args.prefix)
            raw_prompts.append((prompt, mode)); golds.append(gold); full_texts.append(text)

    batch_preds = []
    for p,_ in raw_prompts:
        if args.choice_mode == "rank":
            batch_preds.append(rank_choices(arch, p, choices, score_norm=args.score_norm))
        elif args.choice_mode == "constrained":
            batch_preds.append(constrained_first_token(arch, p, choices))
        else:
            batch_preds.append(greedy_decode(arch, p, max_new=args.max_new))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        for text, (prompt, _), pred, gold in zip(full_texts, raw_prompts, batch_preds, golds):
            w.write(json.dumps({"text": text, "prompt": prompt, "prediction": pred, "gold": gold}, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {args.out}")

if __name__ == "__main__":
    main()


