#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone sanity test for "edge (communication adapter)" training.
- No project imports. Single file.
- Phase-1: freeze vertices (embedding/core/lm_head), train only edges.
- Shows: pre/post val loss, and generation with/without edges.

Requirements: pip install torch transformers
"""

from __future__ import annotations
import argparse, os, math, random, time
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Tiny synthetic corpus
# -----------------------------
PROMPTS_BUY = [
    "beats guidance", "record revenue", "strong demand", "upgraded outlook",
    "accelerating growth", "surging orders", "margin expansion"
]
PROMPTS_SELL = [
    "misses estimates", "guidance cut", "demand weakness", "downgraded",
    "declining growth", "order slowdown", "margin compression"
]
PROMPTS_HOLD = [
    "mixed results", "inline with estimates", "stable outlook", "flat growth",
    "balanced demand", "neutral sentiment", "unchanged guidance"
]

def _make_text(ctx: str, label: str) -> str:
    return f"Context: {ctx}\nTask: Decide among [BUY, HOLD, SELL].\nDecision: {label}\n"

def synth_corpus(n: int, seed: int = 42) -> List[str]:
    random.seed(seed)
    rows = []
    buckets = [
        (PROMPTS_BUY, "BUY"),
        (PROMPTS_SELL, "SELL"),
        (PROMPTS_HOLD, "HOLD"),
    ]
    for _ in range(n):
        pool, lab = random.choice(buckets)
        sig = random.choice(pool)
        ticker = random.choice(["Nvidia", "Apple", "Tesla", "AMD", "Microsoft", "Meta"])
        sector = random.choice(["Semis", "Cloud", "Autos", "AI", "Consumer", "Enterprise"])
        ctx = f"{ticker} ({sector}) {sig}"
        rows.append(_make_text(ctx, lab))
    return rows

class TinyLMDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return {"text": self.texts[i]}

def make_collate(tokenizer, max_len=128):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    def _fn(batch: List[Dict]):
        texts = [b["text"] for b in batch]
        out = tokenizer(texts, max_length=max_len, truncation=True, padding=True, return_tensors="pt")
        ids = out["input_ids"]
        attn = out["attention_mask"]
        labels = ids.clone()  # shift happens in loss
        return {"input_ids": ids, "labels": labels, "attention_mask": attn}
    return _fn

# -----------------------------
# Minimal edge-enabled graph (standalone)
# -----------------------------
class HFCoreWrapper(nn.Module):
    def __init__(self, causal_lm):
        super().__init__()
        self.inner = causal_lm
        self.transformer = getattr(causal_lm, "model", None) or getattr(causal_lm, "transformer")
        if self.transformer is None:
            raise RuntimeError("Unsupported model: cannot find .model/.transformer")
        self.embed_tokens = getattr(self.transformer, "embed_tokens", None)
        if self.embed_tokens is None:
            raise RuntimeError("Unsupported model: cannot find embed_tokens")
        self.lm_head = getattr(causal_lm, "lm_head", None)
        if self.lm_head is None:
            raise RuntimeError("Unsupported model: cannot find lm_head")
        self.hidden_size = int(self.transformer.config.hidden_size)

    @torch.no_grad()
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def core_forward(self, hidden_states, attention_mask=None, position_ids=None):
        out = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
        )
        return out.last_hidden_state

    def logits(self, hidden_states):
        return self.lm_head(hidden_states)

class EdgeAdapter(nn.Module):
    """Linear (src→dst) + 1-layer TransformerEncoder + LayerNorm, causal mask."""
    def __init__(self, src_dim, dst_dim, nhead=8, ffn_mult=4, dropout=0.0, activation="gelu"):
        super().__init__()
        self.proj = nn.Linear(src_dim, dst_dim, bias=False) if src_dim != dst_dim else nn.Identity()
        self.enc = nn.TransformerEncoderLayer(
            d_model=dst_dim, nhead=nhead, dim_feedforward=dst_dim*ffn_mult,
            dropout=dropout, batch_first=True, activation=activation, norm_first=True
        )
        self.norm = nn.LayerNorm(dst_dim)

    def _causal_bool(self, T, device):
        # bool 型因果 mask（True=屏蔽上三角），与 key_padding_mask 同类型，避免 warning
        return torch.ones((T, T), dtype=torch.bool, device=device).triu(1)

    def _kpm(self, attention_mask):
        if attention_mask is None: return None
        if attention_mask.dim() == 4:
            attention_mask = attention_mask.squeeze(1).squeeze(1)
        return (attention_mask == 0)

    def forward(self, src_hidden, attention_mask=None):
        # lazy move to input device/dtype (works with device_map="auto")
        dev, dt = src_hidden.device, src_hidden.dtype
        if next(self.parameters(), None) is not None:
            p = next(self.parameters())
            if p.device != dev or p.dtype != dt:
                self.to(device=dev, dtype=dt)
        x = self.proj(src_hidden)
        T = x.size(1)
        # 统一用 bool mask（positional args 兼容所有 PyTorch 版本）
        x = self.enc(x, self._causal_bool(T, x.device), self._kpm(attention_mask))
        return self.norm(x)

class EmbeddingNode(nn.Module):
    def __init__(self, core: HFCoreWrapper):
        super().__init__()
        self.core = core
        self.out_norm = nn.LayerNorm(core.hidden_size)
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        x = self.core.embed(input_ids)
        if self.out_norm.weight.device != x.device or self.out_norm.weight.dtype != x.dtype:
            self.out_norm = self.out_norm.to(device=x.device, dtype=x.dtype)
        return self.out_norm(x)

class AgentNode(nn.Module):
    def __init__(self, core: HFCoreWrapper):
        super().__init__()
        self.core = core
        self.input_norm = nn.LayerNorm(core.hidden_size)
    def forward(self, incoming, attention_mask=None, position_ids=None):
        agg = incoming if isinstance(incoming, torch.Tensor) else torch.stack(incoming, 0).sum(0)
        if self.input_norm.weight.device != agg.device or self.input_norm.weight.dtype != agg.dtype:
            self.input_norm = self.input_norm.to(device=agg.device, dtype=agg.dtype)
        residual = agg
        x = self.input_norm(agg)
        x = self.core.core_forward(x, attention_mask=attention_mask, position_ids=position_ids)
        return x + residual

class LMHeadNode(nn.Module):
    def __init__(self, core: HFCoreWrapper):
        super().__init__()
        self.core = core
    def forward(self, hidden_states):
        return self.core.logits(hidden_states)

class MiniEdgeGraph(nn.Module):
    def __init__(self, model_id="Qwen/Qwen2-1.5B", torch_dtype=torch.bfloat16, device_map="auto",
                 middle_nodes=2, edge_nhead=8, edge_ffn_mult=4, edge_dropout=0.0, use_edges=True):
        super().__init__()
        self.use_edges = use_edges
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True
        )
        self.core = HFCoreWrapper(self.base)
        h = self.core.hidden_size
        self.src = EmbeddingNode(self.core)
        self.mids = nn.ModuleList([AgentNode(self.core) for _ in range(middle_nodes)])
        self.dst = LMHeadNode(self.core)
        self.edges_s2m = nn.ModuleList([EdgeAdapter(h, h, nhead=edge_nhead, ffn_mult=edge_ffn_mult, dropout=edge_dropout) for _ in range(middle_nodes)])
        self.edges_m2d = nn.ModuleList([EdgeAdapter(h, h, nhead=edge_nhead, ffn_mult=edge_ffn_mult, dropout=edge_dropout) for _ in range(middle_nodes)])
        self._place_custom_modules()

    def _place_custom_modules(self):
        dev = self.core.embed_tokens.weight.device
        dt  = self.core.embed_tokens.weight.dtype
        self.src.to(device=dev, dtype=dt)
        for m in self.mids: m.to(device=dev, dtype=dt)
        for e in list(self.edges_s2m) + list(self.edges_m2d): e.to(device=dev, dtype=dt)

    def freeze_vertices(self):
        for p in self.core.parameters():
            p.requires_grad_(False)
    def edges_parameters(self):
        for m in list(self.edges_s2m) + list(self.edges_m2d):
            for p in m.parameters():
                if p.requires_grad: yield p

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        h0 = self.src(input_ids, attention_mask, position_ids)
        mids_out = []
        for i, node in enumerate(self.mids):
            hin = self.edges_s2m[i](h0, attention_mask) if self.use_edges else h0
            mids_out.append(node(hin, attention_mask, position_ids))
        to_dst = [self.edges_m2d[i](m, attention_mask) if self.use_edges else m for i, m in enumerate(mids_out)]
        hdst = to_dst[0] if len(to_dst)==1 else torch.stack(to_dst, 0).sum(0)
        logits = self.dst(hdst)
        return {"logits": logits}

    def compute_lm_loss(self, input_ids, labels, attention_mask=None, ignore_index=None):
        if ignore_index is None: ignore_index = self.tokenizer.pad_token_id
        out = self.forward(input_ids, attention_mask)
        logits = out["logits"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss, logits

    @torch.no_grad()
    def greedy_decode(self, input_ids, max_new_tokens=16, attention_mask=None):
        self.eval()
        device = input_ids.device
        for _ in range(max_new_tokens):
            out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            next_tok = out["logits"][:, -1, :].argmax(-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_tok.to(device)], dim=1)
            if attention_mask is not None:
                pad = torch.ones((attention_mask.size(0),1), dtype=attention_mask.dtype, device=device)
                attention_mask = torch.cat([attention_mask, pad], dim=1)
        return input_ids

# -----------------------------
# Train/Eval helpers
# -----------------------------
def eval_loss(model: MiniEdgeGraph, dl: DataLoader, ignore_index: int) -> float:
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for b in dl:
            ids = b["input_ids"].to("cuda")
            lbl = b["labels"].to("cuda")
            att = b["attention_mask"].to("cuda")
            loss, _ = model.compute_lm_loss(ids, lbl, attention_mask=att, ignore_index=ignore_index)
            n_tok = (lbl[:, 1:] != ignore_index).sum().item()
            tot_loss += loss.item() * n_tok
            tot_tok  += n_tok
    model.train()
    return tot_loss / max(1, tot_tok)

def sample_decisions(model: MiniEdgeGraph, prompts: List[str], max_new=8) -> List[str]:
    tok = model.tokenizer
    outs = []
    for p in prompts:
        enc = tok(p, return_tensors="pt")
        ids, att = enc["input_ids"].to("cuda"), enc["attention_mask"].to("cuda")
        out_ids = model.greedy_decode(ids, max_new_tokens=max_new, attention_mask=att)
        txt = tok.decode(out_ids[0], skip_special_tokens=True)
        outs.append(txt.split("Decision:")[-1].strip())
    return outs

def count_params_edge_vertex(model: MiniEdgeGraph) -> Tuple[int, int]:
    edge, vert = 0, 0
    edge_param_ids = set()
    for m in list(model.edges_s2m) + list(model.edges_m2d):
        for p in m.parameters(recurse=True):
            edge += p.numel(); edge_param_ids.add(id(p))
    for p in model.parameters():
        if id(p) not in edge_param_ids: vert += p.numel()
    return edge, vert

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2-1.5B", help="Use Qwen/Qwen3-8B if you have more VRAM")
    ap.add_argument("--middle-nodes", type=int, default=2)
    ap.add_argument("--dtype", choices=["bf16","fp16","fp32"], default="bf16")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--save", type=str, default=None, help="Optional: save checkpoint under this dir")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[warn] CUDA not available; this will be very slow on CPU.")

    # seeds
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # dtype
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # model
    model = MiniEdgeGraph(
        model_id=args.model_id, torch_dtype=torch_dtype, device_map="auto",
        middle_nodes=args.middle_nodes, use_edges=True
    )

    tok = model.tokenizer
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    # data
    train_texts = synth_corpus(300, seed=args.seed)
    val_texts   = synth_corpus(90,  seed=args.seed+1)
    collate = make_collate(tok, max_len=args.max_len)
    dl_train = DataLoader(TinyLMDataset(train_texts), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, collate_fn=collate)
    dl_val   = DataLoader(TinyLMDataset(val_texts),   batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0, collate_fn=collate)

    # freeze vertices, train edges
    model.freeze_vertices()
    edge_params = list(model.edges_parameters())
    optimizer = torch.optim.AdamW(edge_params, lr=args.lr, weight_decay=0.01)

    # amp (new API to avoid deprecation)
    use_fp16 = (args.dtype == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    autocast_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if args.dtype=="bf16" else None)

    # report
    e_cnt, v_cnt = count_params_edge_vertex(model)
    base = eval_loss(model, dl_val, ignore_index=tok.pad_token_id)
    print(f"\n[info] model={args.model_id}  middle_nodes={args.middle_nodes}  dtype={args.dtype}")
    print(f"[info] edge params: {e_cnt/1e6:.2f}M | vertex params (frozen): {v_cnt/1e6:.2f}M")
    print(f"[eval:before] val loss = {base:.4f}  ppl = {math.exp(base):.2f}")

    # train few steps
    model.train()
    step, t0 = 0, time.time()
    while step < args.steps:
        for b in dl_train:
            step += 1
            ids = b["input_ids"].to("cuda")
            lbl = b["labels"].to("cuda")
            att = b["attention_mask"].to("cuda")

            if autocast_dtype is not None:
                ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype)
            else:
                class _NoOp:
                    def __enter__(self): pass
                    def __exit__(self, *exc): return False
                ctx = _NoOp()

            with ctx:
                loss, _ = model.compute_lm_loss(ids, lbl, attention_mask=att, ignore_index=tok.pad_token_id)

            if use_fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(edge_params, 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(edge_params, 1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0:
                dt = time.time() - t0
                print(f"[train] step {step:04d}  loss={loss.item():.4f}  ({dt:.1f}s/10 steps)")
                t0 = time.time()
            if step >= args.steps: break

    after = eval_loss(model, dl_val, ignore_index=tok.pad_token_id)
    print(f"[eval:after ] val loss = {after:.4f}  ppl = {math.exp(after):.2f}")
    print(f"[delta] loss change = {base - after:+.4f}  (positive => improved)")

    # generations: with vs without edges
    probes = [
        "Context: Nvidia record revenue and upgraded outlook\nTask: Decide among [BUY, HOLD, SELL].\nDecision:",
        "Context: Tesla guidance cut with margin compression\nTask: Decide among [BUY, HOLD, SELL].\nDecision:",
        "Context: Microsoft inline with estimates and stable outlook\nTask: Decide among [BUY, HOLD, SELL].\nDecision:",
    ]
    print("\n[gen] with trained edges:")
    model.use_edges = True
    for s in sample_decisions(model, probes, max_new=8):
        print("  ", s)

    print("\n[gen] bypass edges (baseline):")
    model.use_edges = False
    for s in sample_decisions(model, probes, max_new=8):
        print("  ", s)

    # optional save
    if args.save:
        step_dir = os.path.join(args.save, f"step_{args.steps}")
        os.makedirs(step_dir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "meta": {"steps": args.steps}}, os.path.join(step_dir, "ckpt.pt"))
        print(f"[ok] saved to {step_dir}/ckpt.pt")

if __name__ == "__main__":
    main()


