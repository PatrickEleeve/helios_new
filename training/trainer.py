# training/trainer.py
from __future__ import annotations
import os
import math
import json
import time
import random
import glob
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Iterable

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from training.eval_metrics import parse_decision

# 依赖你的架构实现
from helios.architecture import HeliosArchitecture, DEFAULT_MODEL_ID
from helios.protocols import FlowConfig, DebateConfig, DenseMode

# -----------------------------
# 数据集与打包
# -----------------------------
class LMDataset(Dataset):
    def __init__(self, texts: List[str]): super().__init__(); self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return {"text": self.texts[idx]}

@dataclass
class CollateConfig:
    tokenizer: AutoTokenizer
    max_seq_len: int = 1024
    add_bos: bool = True
    add_eos: bool = True

class LMCollateFn:
    def __init__(self, cfg: CollateConfig):
        self.tok = cfg.tokenizer; self.max_seq_len = cfg.max_seq_len
        self.add_bos = cfg.add_bos; self.add_eos = cfg.add_eos
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
    def __call__(self, batch: List[Dict]):
        texts = [ex["text"] for ex in batch]
        if self.add_bos and self.tok.bos_token: texts = [self.tok.bos_token + t for t in texts]
        if self.add_eos and self.tok.eos_token: texts = [t + self.tok.eos_token for t in texts]
        out = self.tok(texts, max_length=self.max_seq_len, truncation=True, padding=True, return_tensors="pt")
        input_ids = out["input_ids"]; attention_mask = out["attention_mask"]; labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

# -----------------------------
# 训练配置
# -----------------------------
@dataclass
class TrainConfig:
    # 模型与数据
    model_id: str = DEFAULT_MODEL_ID
    max_seq_len: int = 1024
    middle_nodes: int = 2  # 可选: 仅用于某些简化/对比实验；HeliosArchitecture 当前未使用

    # 训练
    epochs: int = 1
    train_bs: int = 4
    eval_bs: int = 4
    lr_edges: float = 1e-4
    lr_vertices: float = 5e-6
    weight_decay: float = 0.01
    grad_accum: int = 1
    warmup_ratio: float = 0.03
    max_steps: Optional[int] = None
    phase1_edges_only: bool = True
    phase2_unfreeze_at_epoch: Optional[int] = None

    # 精度与设备
    dtype: str = "bf16"              # "bf16" | "fp16" | "fp32"
    device: str = "cuda"
    compile: bool = False

    # 日志/保存
    log_every: int = 20
    eval_every: int = 200
    save_every: int = 1000
    out_dir: str = "outputs/exp_default"
    save_max_keep: int = 3

    # 其他
    seed: int = 42
    trust_remote_code: bool = True

    # 断点续训
    resume_from: Optional[str] = None
    save_optimizer: bool = True

    # TradingAgents 流程参数（可选）
    dense_mode: str = "dense"         # "dense"|"hybrid"
    research_rounds: int = 3
    risk_rounds: int = 3
    use_edges: bool = True
    # 内存优化
    gradient_checkpointing: bool = True
    # 稳定性与损失
    edge_identity_init: bool = False
    label_smoothing: float = 0.0
    # 决策排名评估（可选）
    eval_rank_choices: Optional[str] = None  # 逗号分隔，如 "BUY,HOLD,SELL"
    eval_rank_prefix: str = "Decision:"
    eval_rank_length_norm: bool = False

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _dtype_from_str(x: str):
    x = x.lower()
    if x == "bf16": return torch.bfloat16
    if x == "fp16": return torch.float16
    if x == "fp32": return torch.float32
    raise ValueError(f"Unsupported dtype: {x}")

def load_texts_from_file(path: str, field: Optional[str] = None) -> List[str]:
    texts: List[str] = []
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line: texts.append(line)
        return texts
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                key = field or ("text" if "text" in obj else "prompt")
                texts.append(str(obj[key]))
        return texts
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for x in obj:
                key = field or ("text" if isinstance(x, dict) and "text" in x else "prompt")
                texts.append(str(x[key]) if isinstance(x, dict) else str(x))
        elif isinstance(obj, dict):
            key = field or ("text" if "text" in obj else "prompt")
            arr = obj.get(key, None)
            if isinstance(arr, list): texts.extend([str(t) for t in arr])
            else: texts.append(json.dumps(obj, ensure_ascii=False))
        else:
            texts.append(str(obj))
        return texts
    raise ValueError(f"Unsupported file format: {path}")

def _resolve_ckpt_path(p: str) -> str:
    if os.path.isdir(p):
        cand = sorted(glob.glob(os.path.join(p, "step_*", "ckpt.pt")))
        if cand: return cand[-1]
        pt = os.path.join(p, "ckpt.pt")
        if os.path.exists(pt): return pt
        raise FileNotFoundError(f"No ckpt.pt found under: {p}")
    else:
        if not os.path.exists(p): raise FileNotFoundError(p)
        return p

class CheckpointManager:
    def __init__(self, out_dir: str, max_keep: int = 3):
        os.makedirs(out_dir, exist_ok=True); self.out_dir = out_dir
        self.max_keep = max_keep; self.saved: List[str] = []
    def save(self, payload: Dict, step: int):
        path = os.path.join(self.out_dir, f"step_{step}"); os.makedirs(path, exist_ok=True)
        torch.save(payload, os.path.join(path, "ckpt.pt")); self.saved.append(path)
        while len(self.saved) > self.max_keep:
            old = self.saved.pop(0)
            try:
                for fn in os.listdir(old): os.remove(os.path.join(old, fn))
                os.rmdir(old)
            except Exception: pass

class Trainer:
    def __init__(self, cfg: TrainConfig, train_texts: List[str], eval_texts: Optional[List[str]] = None):
        self.cfg = cfg; set_seed(cfg.seed)
        dtype = _dtype_from_str(cfg.dtype)

        flow = FlowConfig(
            dense_mode=DenseMode.DENSE if cfg.dense_mode == "dense" else DenseMode.HYBRID,
            use_edges=cfg.use_edges,
            debate=DebateConfig(n_research_rounds=cfg.research_rounds, n_risk_rounds=cfg.risk_rounds),
        )
        self.arch = HeliosArchitecture(
            model_id=cfg.model_id,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            flow=flow,
            gradient_checkpointing=cfg.gradient_checkpointing,
            edge_dropout=0.0,
            edge_identity_init=cfg.edge_identity_init,
        )
        self.tok = self.arch.tokenizer

        collate = LMCollateFn(CollateConfig(tokenizer=self.tok, max_seq_len=cfg.max_seq_len))
        self.train_dl = DataLoader(LMDataset(train_texts), batch_size=cfg.train_bs, shuffle=True, drop_last=True, num_workers=2, collate_fn=collate)
        self.eval_dl = None
        if eval_texts:
            self.eval_dl = DataLoader(LMDataset(eval_texts), batch_size=cfg.eval_bs, shuffle=False, drop_last=False, num_workers=2, collate_fn=collate)
        self._eval_texts: Optional[List[str]] = eval_texts

        self._optim_mode = "edges_only" if cfg.phase1_edges_only else "mixed"
        self.optimizer = self._build_optimizer(self._optim_mode)
        total_steps = cfg.max_steps or (len(self.train_dl) // cfg.grad_accum) * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == "fp16"))
        self.autocast_dtype = torch.float16 if cfg.dtype == "fp16" else (torch.bfloat16 if cfg.dtype == "bf16" else None)

        if cfg.compile and hasattr(torch, "compile"):
            self.arch = torch.compile(self.arch)  # type: ignore

        self.ckpt = CheckpointManager(cfg.out_dir, cfg.save_max_keep)
        self._epoch = 0; self._global_step_for_accum = 0; self._train_steps_completed = 0; self._phase2_done = False

        if cfg.resume_from:
            self._resume_from(cfg.resume_from)

    def _build_optimizer(self, mode: str) -> torch.optim.Optimizer:
        cfg = self.cfg
        if mode == "edges_only":
            self.arch.freeze_vertices()
            params = list(self.arch.edges_parameters())
            return torch.optim.AdamW(params, lr=cfg.lr_edges, weight_decay=cfg.weight_decay)
        else:
            self.arch.unfreeze_vertices()
            edge_params = list(self.arch.edges_parameters())
            vert_params = list(self.arch.vertices_parameters())
            return torch.optim.AdamW(
                [{"params": edge_params, "lr": cfg.lr_edges, "weight_decay": cfg.weight_decay},
                 {"params": vert_params, "lr": cfg.lr_vertices, "weight_decay": cfg.weight_decay}]
            )

    def _resume_from(self, resume_from: str):
        path = _resolve_ckpt_path(resume_from)
        state = torch.load(path, map_location="cpu")
        arch_sd = state.get("arch") or state.get("state_dict")
        if arch_sd is None: raise RuntimeError("Checkpoint missing 'arch'/'state_dict'")
        self.arch.load_state_dict(arch_sd, strict=False)
        meta = state.get("meta", {})
        self._epoch = int(meta.get("epoch", 0))
        self._train_steps_completed = int(meta.get("train_steps_completed", 0))
        self._global_step_for_accum = int(meta.get("global_step_for_accum", 0))
        self._phase2_done = bool(meta.get("phase2_done", False))
        self._optim_mode = meta.get("optim_mode", self._optim_mode)

        self.optimizer = self._build_optimizer(self._optim_mode)
        cfg = self.cfg
        total_steps = cfg.max_steps or (len(self.train_dl) // cfg.grad_accum) * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        if cfg.save_optimizer:
            opt_sd = state.get("optimizer");  sch_sd = state.get("scheduler"); scl_sd = state.get("scaler")
            if opt_sd:
                try: self.optimizer.load_state_dict(opt_sd)
                except Exception as e: print(f"[warn] optimizer.load_state_dict failed: {e}")
            if sch_sd:
                try: self.scheduler.load_state_dict(sch_sd)
                except Exception as e: print(f"[warn] scheduler.load_state_dict failed: {e}")
            if scl_sd and self.cfg.dtype == "fp16":
                try: self.scaler.load_state_dict(scl_sd)
                except Exception as e: print(f"[warn] scaler.load_state_dict failed: {e}")
        print(f"[resume] from {path}  | epoch={self._epoch} steps_done={self._train_steps_completed} mode={self._optim_mode}")

    def _make_payload(self, step: int, extra: Optional[Dict] = None) -> Dict:
        payload = {
            "arch": self.arch.state_dict(),
            "cfg": asdict(self.cfg),
            "meta": {
                "epoch": self._epoch,
                "train_steps_completed": self._train_steps_completed,
                "global_step_for_accum": self._global_step_for_accum,
                "phase2_done": self._phase2_done,
                "optim_mode": self._optim_mode,
                **(extra or {}),
            },
        }
        if self.cfg.save_optimizer:
            payload["optimizer"] = self.optimizer.state_dict()
            payload["scheduler"] = self.scheduler.state_dict()
            if self.cfg.dtype == "fp16":
                payload["scaler"] = self.scaler.state_dict()
        return payload

    def train(self):
        cfg = self.cfg
        step = self._train_steps_completed
        best_eval = float("inf")

        self.arch.train(); start_ts = time.time(); running_loss = 0.0
        for epoch in range(self._epoch, cfg.epochs):
            self._epoch = epoch

            if cfg.phase2_unfreeze_at_epoch is not None and epoch >= cfg.phase2_unfreeze_at_epoch and not self._phase2_done:
                self._optim_mode = "mixed"
                self.optimizer = self._build_optimizer(self._optim_mode)
                total_steps = cfg.max_steps or (len(self.train_dl) // cfg.grad_accum) * (cfg.epochs - epoch)
                warmup_steps = int(total_steps * cfg.warmup_ratio)
                self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
                self._phase2_done = True
                print(f"[phase2] unfreeze vertices at epoch {epoch}")

            for it, batch in enumerate(self.train_dl):
                step += 1
                loss = self._train_step(batch)
                running_loss += loss
                if step % cfg.log_every == 0:
                    avg = running_loss / cfg.log_every
                    tps = cfg.train_bs * cfg.grad_accum / max(1e-6, (time.time() - start_ts))
                    print(f"[epoch {epoch} | step {step}] loss={avg:.4f}  ~{tps:.1f} samples/s")
                    running_loss = 0.0; start_ts = time.time()

                if self.eval_dl is not None and step % cfg.eval_every == 0:
                    val = self.evaluate()
                    print(f"[eval] step {step} ppl={math.exp(val):.2f} loss={val:.4f}")
                    if self._eval_texts is not None and self.cfg.eval_rank_choices:
                        try:
                            acc = self.evaluate_rank()
                            print(f"[eval|rank] step {step} acc={acc:.3f} choices={self.cfg.eval_rank_choices}")
                        except Exception as e:
                            print(f"[warn] rank eval failed: {e}")
                    if val < best_eval:
                        best_eval = val
                        self.ckpt.save(self._make_payload(step, {"val_loss": val}), step)

                if step % cfg.save_every == 0:
                    self.ckpt.save(self._make_payload(step, {"note": "periodic"}), step)

                self._train_steps_completed = step
                if cfg.max_steps is not None and step >= cfg.max_steps: break
            if cfg.max_steps is not None and step >= cfg.max_steps: break
        self.ckpt.save(self._make_payload(step, {"final": True}), step)

    def _train_step(self, batch: Dict) -> float:
        cfg = self.cfg
        input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
        labels = batch["labels"].to(cfg.device, non_blocking=True)
        attn = batch.get("attention_mask", None)
        if attn is not None: attn = attn.to(cfg.device, non_blocking=True)

        if self.autocast_dtype is not None:
            ctx = torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
        else:
            class _NoOp: 
                def __enter__(self): pass
                def __exit__(self, *exc): return False
            ctx = _NoOp()

        with ctx:
            loss, _ = self.arch.compute_lm_loss(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attn,
                ignore_index=self.tok.pad_token_id,
                label_smoothing=self.cfg.label_smoothing,
            )
            loss = loss / self.cfg.grad_accum

        if self.cfg.dtype == "fp16":
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self._global_step_for_accum += 1
        if (self._global_step_for_accum % self.cfg.grad_accum) == 0:
            if self.cfg.dtype == "fp16":
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self._all_trainable_params(), 1.0)
                self.scaler.step(self.optimizer); self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self._all_trainable_params(), 1.0)
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True); self.scheduler.step()

        return loss.item() * self.cfg.grad_accum

    def _all_trainable_params(self) -> Iterable[torch.nn.Parameter]:
        return (p for p in self.arch.parameters() if p.requires_grad)

    @torch.no_grad()
    def evaluate(self) -> float:
        cfg = self.cfg; self.arch.eval()
        total_loss, total_tokens = 0.0, 0
        for batch in self.eval_dl:  # type: ignore
            input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
            labels = batch["labels"].to(cfg.device, non_blocking=True)
            attn = batch.get("attention_mask", None)
            if attn is not None: attn = attn.to(cfg.device, non_blocking=True)
            loss, _ = self.arch.compute_lm_loss(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attn,
                ignore_index=self.arch.tokenizer.pad_token_id,
                label_smoothing=self.cfg.label_smoothing,
            )
            n_tok = (labels[:, 1:] != self.arch.tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * n_tok; total_tokens += n_tok
        self.arch.train()
        if total_tokens == 0: return float("inf")
        return total_loss / total_tokens

    @torch.no_grad()
    def evaluate_rank(self) -> float:
        assert self._eval_texts is not None
        choices = [c.strip() for c in (self.cfg.eval_rank_choices or "").split(",") if c.strip()]
        if not choices:
            return float('nan')
        device = self.cfg.device
        self.arch.eval()
        correct = 0
        total = 0
        tok = self.tok
        for text in self._eval_texts:
            gold = parse_decision(text)
            if gold not in choices:
                continue
            if self.cfg.eval_rank_prefix in text:
                prompt = text.split(self.cfg.eval_rank_prefix, 1)[0] + self.cfg.eval_rank_prefix
            else:
                prompt = text
            base = tok(prompt, return_tensors="pt")
            base_ids = base["input_ids"].to(device)
            base_attn = base["attention_mask"].to(device)
            scores = []
            for ch in choices:
                ch_ids = tok(" " + ch, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
                ctx_ids = base_ids.clone(); ctx_attn = base_attn.clone()
                logp = 0.0
                for tid in ch_ids:
                    out = self.arch.forward(input_ids=ctx_ids, attention_mask=ctx_attn)["logits"]
                    lp = torch.log_softmax(out[:, -1, :], dim=-1)
                    logp += lp[0, int(tid)].item()
                    tid_ = tid.view(1,1)
                    ctx_ids = torch.cat([ctx_ids, tid_.to(ctx_ids.device)], dim=1)
                    one = torch.ones((ctx_attn.size(0),1), dtype=ctx_attn.dtype, device=ctx_attn.device)
                    ctx_attn = torch.cat([ctx_attn, one], dim=1)
                if self.cfg.eval_rank_length_norm and len(ch_ids) > 0:
                    logp = logp / len(ch_ids)
                scores.append(logp)
            pred = choices[int(torch.tensor(scores).argmax().item())]
            correct += int(pred == gold)
            total += 1
        self.arch.train()
        return correct / max(1, total)

def run_training(train_file: str, eval_file: Optional[str] = None, field: Optional[str] = None, cfg: Optional[TrainConfig] = None):
    cfg = cfg or TrainConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)
    train_texts = load_texts_from_file(train_file, field=field)
    eval_texts = load_texts_from_file(eval_file, field=field) if eval_file else None
    print(f"[data] train={len(train_texts)}  eval={0 if eval_texts is None else len(eval_texts)}")
    print(f"[cfg] {cfg}")
    trainer = Trainer(cfg, train_texts, eval_texts)
    trainer.train()

def main(train_cfg: TrainConfig, helios_cfg, train_files: List[str], val_files: List[str]):
    """主训练函数

    - 汇总多个训练/验证文件的数据
    - 构建 Trainer 并启动两阶段训练（先边后顶点）

    说明：helios_cfg 参数当前未使用（Trainer 会基于 TrainConfig 自行构造 FlowConfig），
    为向后兼容保留该参数位。
    """
    os.makedirs(train_cfg.out_dir, exist_ok=True)

    # 读取并汇总数据
    train_texts: List[str] = []
    for fp in (train_files or []):
        try:
            train_texts.extend(load_texts_from_file(fp))
        except Exception as e:
            print(f"[warn] failed to load train file {fp}: {e}")

    eval_texts: Optional[List[str]] = []
    for fp in (val_files or []):
        try:
            eval_texts.extend(load_texts_from_file(fp))
        except Exception as e:
            print(f"[warn] failed to load val file {fp}: {e}")
    if eval_texts == []:
        eval_texts = None

    print(f"[data] train={len(train_texts)}  eval={0 if eval_texts is None else len(eval_texts)}")
    print(f"[cfg] {train_cfg}")

    trainer = Trainer(train_cfg, train_texts, eval_texts)
    trainer.train()
