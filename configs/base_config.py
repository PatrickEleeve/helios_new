# configs/base_config.py
from dataclasses import asdict
from pathlib import Path
from training.trainer import TrainConfig

# 项目根目录（相对本文件）
ROOT = Path(__file__).resolve().parents[1]

# 通用路径
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 默认底模与训练输出
DEFAULT_MODEL_ID = str(ROOT / "models" / "Qwen3-4B")  # 可替换为其他模型
DEFAULT_OUT_DIR = str(ROOT / "outputs" / "exp_default")

# 全局缺省
GLOBAL_DEFAULTS = dict(
    model_id=DEFAULT_MODEL_ID,
    middle_nodes=2,
    max_seq_len=1024,
    epochs=1,
    train_bs=4,
    eval_bs=4,
    lr_edges=1e-4,
    lr_vertices=5e-6,
    weight_decay=0.01,
    grad_accum=2,
    warmup_ratio=0.03,
    max_steps=None,
    phase1_edges_only=True,
    phase2_unfreeze_at_epoch=None,
    dtype="bf16",
    device="cuda",
    compile=False,
    log_every=20,
    eval_every=200,
    save_every=1000,
    out_dir=DEFAULT_OUT_DIR,
    save_max_keep=3,
    seed=42,
    trust_remote_code=True,
)

def make_train_config(**overrides) -> TrainConfig:
    """构造一个 TrainConfig，可在实验里覆盖字段"""
    data = {**GLOBAL_DEFAULTS, **overrides}
    return TrainConfig(**data)  # type: ignore