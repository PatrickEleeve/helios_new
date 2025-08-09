# training/trainer.py
from __future__ import annotations
import os
import math
import json
import time
import random
import glob
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Iterable, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# 依赖你的架构实现
from helios.architecture import HeliosArchitecture, DEFAULT_MODEL_ID


# -----------------------------
# 数据集与打包
# -----------------------------
class LMDataset(Dataset):
    """
    最小LM数据集：接受纯文本样本或 JSON/JSONL 中的字段。
    """
    def __init__(self, texts: List[str]):
        super().__init__()
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx]}


@dataclass
class CollateConfig:
    tokenizer: AutoTokenizer
    max_seq_len: int = 1024
    add_bos: bool = True
    add_eos: bool = True


class LMCollateFn:
    """
    把 raw text -> input_ids/labels/attention_mask
    自回归：labels = input_ids 的右移一位（在 Trainer 里做）
    """
    def __init__(self, cfg: CollateConfig):
        self.tok = cfg.tokenizer
        self.max_seq_len = cfg.max_seq_len
        self.add_bos = cfg.add_bos
        self.add_eos = cfg.add_eos
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, batch: List[Dict]):
        texts = [ex["text"] for ex in batch]

        if self.add_bos and self.tok.bos_token:
            texts = [self.tok.bos_token + t for t in texts]
        if self.add_eos and self.tok.eos_token:
            texts = [t + self.tok.eos_token for t in texts]

        out = self.tok(
            texts,
            max_length=self.max_seq_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = out["input_ids"]
        attention_mask = out["attention_mask"]
        labels = input_ids.clone()  # shift 在 compute_lm_loss 里做

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# -----------------------------
# 训练配置
# -----------------------------
@dataclass
class TrainConfig:
    # 模型与数据
    model_id: str = DEFAULT_MODEL_ID
    middle_nodes: int = 2
    max_seq_len: int = 1024

    # 训练
    epochs: int = 1
    train_bs: int = 4
    eval_bs: int = 4
    lr_edges: float = 1e-4
    lr_vertices: float = 5e-6
    weight_decay: float = 0.01
    grad_accum: int = 1
    warmup_ratio: float = 0.03
    max_steps: Optional[int] = None  # None=按epochs
    phase1_edges_only: bool = True   # True=只训边
    phase2_unfreeze_at_epoch: Optional[int] = None  # None=不解冻

    # 精度与设备
    dtype: str = "bf16"              # "bf16" | "fp16" | "fp32"
    device: str = "cuda"
    compile: bool = False

    # 日志/保存
    log_every: int = 20
    eval_every: int = 200
    save_every: int = 1000
    out_dir: str = "outputs/exp_default"
    save_max_keep: int = 3  # 最多保留最近的N个ckpt

    # 其他
    seed: int = 42
    trust_remote_code: bool = True

    # 断点续训
    resume_from: Optional[str] = None   # 路径可指向 outputs/exp_xxx/ 或 step_xxx/ 或 ckpt.pt
    save_optimizer: bool = True         # 保存优化器/调度器/scaler


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dtype_from_str(x: str):
    x = x.lower()
    if x == "bf16":
        return torch.bfloat16
    if x == "fp16":
        return torch.float16
    if x == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {x}")


def load_texts_from_file(path: str, field: Optional[str] = None) -> List[str]:
    """
    统一加载：支持 .txt（每行一条） / .jsonl（每行一个JSON） / .json（list或dict）
    - field: 若是JSON/JSONL，指定文本字段名；不传则尝试 "text" 或 "prompt"
    """
    texts: List[str] = []
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        return texts

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
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
                if isinstance(x, dict):
                    texts.append(str(x[key]))
                else:
                    texts.append(str(x))
        elif isinstance(obj, dict):
            key = field or ("text" if "text" in obj else "prompt")
            arr = obj.get(key, None)
            if isinstance(arr, list):
                texts.extend([str(t) for t in arr])
            else:
                texts.append(json.dumps(obj, ensure_ascii=False))
        else:
            texts.append(str(obj))
        return texts

    raise ValueError(f"Unsupported file format: {path}")


# -----------------------------
# Checkpoint 管理
# -----------------------------
def _resolve_ckpt_path(p: str) -> str:
    """
    支持：
      - outputs/exp_xxx/
      - outputs/exp_xxx/step_yyy/
      - outputs/exp_xxx/step_yyy/ckpt.pt
      - 任意 *.pt
    返回实际 ckpt.pt 文件路径
    """
    if os.path.isdir(p):
        # 传的是实验目录
        cand = sorted(glob.glob(os.path.join(p, "step_*", "ckpt.pt")))
        if cand:
            return cand[-1]
        # 传的是某个 step 目录
        pt = os.path.join(p, "ckpt.pt")
        if os.path.exists(pt):
            return pt
        raise FileNotFoundError(f"No ckpt.pt found under: {p}")
    else:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p


class CheckpointManager:
    def __init__(self, out_dir: str, max_keep: int = 3):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.max_keep = max_keep
        self.saved: List[str] = []

    def save(self, payload: Dict, step: int):
        path = os.path.join(self.out_dir, f"step_{step}")
        os.makedirs(path, exist_ok=True)
        torch.save(payload, os.path.join(path, "ckpt.pt"))
        self.saved.append(path)
        # 清理
        while len(self.saved) > self.max_keep:
            old = self.saved.pop(0)
            try:
                for fn in os.listdir(old):
                    os.remove(os.path.join(old, fn))
                os.rmdir(old)
            except Exception:
                pass


# -----------------------------
# Trainer
# -----------------------------
class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        train_texts: List[str],
        eval_texts: Optional[List[str]] = None,
    ):
        self.cfg = cfg
        set_seed(cfg.seed)

        # 1) 初始化架构（Qwen3-8B 为默认）
        dtype = _dtype_from_str(cfg.dtype)
        self.arch = HeliosArchitecture(
            model_id=cfg.model_id,
            device_map="auto",
            torch_dtype=dtype,
            middle_nodes=cfg.middle_nodes,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.tok = self.arch.tokenizer

        # 2) DataLoader
        collate = LMCollateFn(CollateConfig(tokenizer=self.tok, max_seq_len=cfg.max_seq_len))
        self.train_dl = DataLoader(
            LMDataset(train_texts),
            batch_size=cfg.train_bs,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            collate_fn=collate,
        )
        self.eval_dl = None
        if eval_texts:
            self.eval_dl = DataLoader(
                LMDataset(eval_texts),
                batch_size=cfg.eval_bs,
                shuffle=False,
                drop_last=False,
                num_workers=2,
                collate_fn=collate,
            )

        # 3) Optimizer & Scheduler 构造（可能在 resume 时被覆盖）
        self._optim_mode = "edges_only" if cfg.phase1_edges_only else "mixed"
        self.optimizer = self._build_optimizer(self._optim_mode)
        total_steps = cfg.max_steps or (len(self.train_dl) // cfg.grad_accum) * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # 4) AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == "fp16"))
        self.autocast_dtype = torch.float16 if cfg.dtype == "fp16" else (torch.bfloat16 if cfg.dtype == "bf16" else None)

        # 5) compile（可选）
        if cfg.compile and hasattr(torch, "compile"):
            self.arch = torch.compile(self.arch)  # type: ignore

        # 6) 断点
        self.ckpt = CheckpointManager(cfg.out_dir, cfg.save_max_keep)
        self._epoch = 0
        self._global_step_for_accum = 0
        self._train_steps_completed = 0
        self._phase2_done = False

        # 7) 恢复
        if cfg.resume_from:
            self._resume_from(cfg.resume_from)

    # ---------- Optimizer 构造 ----------
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
                [
                    {"params": edge_params, "lr": cfg.lr_edges, "weight_decay": cfg.weight_decay},
                    {"params": vert_params, "lr": cfg.lr_vertices, "weight_decay": cfg.weight_decay},
                ]
            )

    # ---------- 断点恢复 ----------
    def _resume_from(self, resume_from: str):
        path = _resolve_ckpt_path(resume_from)
        state = torch.load(path, map_location="cpu")

        # 1) 模型
        arch_sd = state.get("arch") or state.get("state_dict")
        if arch_sd is None:
            raise RuntimeError("Checkpoint missing 'arch'/'state_dict'")
        self.arch.load_state_dict(arch_sd, strict=False)

        # 2) 元信息
        meta = state.get("meta", {})
        self._epoch = int(meta.get("epoch", 0))
        self._train_steps_completed = int(meta.get("train_steps_completed", 0))
        self._global_step_for_accum = int(meta.get("global_step_for_accum", 0))
        self._phase2_done = bool(meta.get("phase2_done", False))
        self._optim_mode = meta.get("optim_mode", self._optim_mode)  # fallback 当前配置

        # 3) 重新按保存时的模式构造优化器/调度器，再加载状态
        self.optimizer = self._build_optimizer(self._optim_mode)

        # scheduler 需要重建同样的 total_steps 再加载
        cfg = self.cfg
        total_steps = cfg.max_steps or (len(self.train_dl) // cfg.grad_accum) * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        if cfg.save_optimizer:
            opt_sd = state.get("optimizer")
            if opt_sd:
                try:
                    self.optimizer.load_state_dict(opt_sd)
                except Exception as e:
                    print(f"[warn] optimizer.load_state_dict failed: {e}")

            sch_sd = state.get("scheduler")
            if sch_sd:
                try:
                    self.scheduler.load_state_dict(sch_sd)
                except Exception as e:
                    print(f"[warn] scheduler.load_state_dict failed: {e}")

            scl_sd = state.get("scaler")
            if scl_sd and self.cfg.dtype == "fp16":
                try:
                    self.scaler.load_state_dict(scl_sd)
                except Exception as e:
                    print(f"[warn] scaler.load_state_dict failed: {e}")

        print(f"[resume] from {path}  | epoch={self._epoch} steps_done={self._train_steps_completed} mode={self._optim_mode}")

    # ---------- 保存 ----------
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

    # ---------- 训练主循环 ----------
    def train(self):
        cfg = self.cfg
        step = self._train_steps_completed  # 从断点继续统计
        best_eval = float("inf")

        self.arch.train()
        start_ts = time.time()
        running_loss = 0.0

        for epoch in range(self._epoch, cfg.epochs):
            self._epoch = epoch

            # Phase-2：按 epoch 解冻顶点
            if cfg.phase2_unfreeze_at_epoch is not None and epoch >= cfg.phase2_unfreeze_at_epoch and not self._phase2_done:
                # 切换到 mixed 优化模式
                self._optim_mode = "mixed"
                self.optimizer = self._build_optimizer(self._optim_mode)
                # 重新构建 scheduler（简单起见，重置；也可用剩余步数重建）
                total_steps = cfg.max_steps or (len(self.train_dl) // cfg.grad_accum) * (cfg.epochs - epoch)
                warmup_steps = int(total_steps * cfg.warmup_ratio)
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
                )
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
                    running_loss = 0.0
                    start_ts = time.time()

                if self.eval_dl is not None and step % cfg.eval_every == 0:
                    val = self.evaluate()
                    print(f"[eval] step {step} ppl={math.exp(val):.2f} loss={val:.4f}")
                    if val < best_eval:
                        best_eval = val
                        self.ckpt.save(self._make_payload(step, {"val_loss": val}), step)

                if step % cfg.save_every == 0:
                    self.ckpt.save(self._make_payload(step, {"note": "periodic"}), step)

                self._train_steps_completed = step

                if cfg.max_steps is not None and step >= cfg.max_steps:
                    break

            if cfg.max_steps is not None and step >= cfg.max_steps:
                break

        # 最后一存
        self.ckpt.save(self._make_payload(step, {"final": True}), step)

    def _train_step(self, batch: Dict) -> float:
        cfg = self.cfg

        input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
        labels = batch["labels"].to(cfg.device, non_blocking=True)
        attn = batch.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(cfg.device, non_blocking=True)

        # AMP autocast
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
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self._all_trainable_params(), 1.0)
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        # 返回未经 /grad_accum 的标量
        return loss.item() * self.cfg.grad_accum

    def _all_trainable_params(self) -> Iterable[torch.nn.Parameter]:
        return (p for p in self.arch.parameters() if p.requires_grad)

    @torch.no_grad()
    def evaluate(self) -> float:
        cfg = self.cfg
        self.arch.eval()
        total_loss, total_tokens = 0.0, 0

        for batch in self.eval_dl:  # type: ignore
            input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
            labels = batch["labels"].to(cfg.device, non_blocking=True)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(cfg.device, non_blocking=True)

            loss, _ = self.arch.compute_lm_loss(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attn,
                ignore_index=self.arch.tokenizer.pad_token_id,
            )
            n_tok = (labels[:, 1:] != self.arch.tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

        self.arch.train()
        if total_tokens == 0:
            return float("inf")
        return total_loss / total_tokens


# -----------------------------
# 便捷入口
# -----------------------------
def run_training(
    train_file: str,
    eval_file: Optional[str] = None,
    field: Optional[str] = None,
    cfg: Optional[TrainConfig] = None,
):
    """
    直接用文件跑：支持 .txt/.json/.jsonl
    - field: JSON/JSONL 的文本字段名；不填则尝试 "text"/"prompt"
    """
    cfg = cfg or TrainConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_texts = load_texts_from_file(train_file, field=field)
    eval_texts = load_texts_from_file(eval_file, field=field) if eval_file else None

    print(f"[data] train={len(train_texts)}  eval={0 if eval_texts is None else len(eval_texts)}")
    print(f"[cfg] {cfg}")

    trainer = Trainer(cfg, train_texts, eval_texts)
    trainer.train()
