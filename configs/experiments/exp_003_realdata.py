# configs/experiments/exp_003_realdata.py
from pathlib import Path
from configs.base_config import make_train_config, PROCESSED_DIR, DEFAULT_MODEL_ID

# 期望的预处理输出文件（见下方“预处理脚本与命令”）
FPB     = str(PROCESSED_DIR / "fpb_train.jsonl")
FIQA    = str(PROCESSED_DIR / "fiqa_train.jsonl")
FINQA   = str(PROCESSED_DIR / "finqa_train.jsonl")
TATQA   = str(PROCESSED_DIR / "tatqa_train.jsonl")
CONV    = str(PROCESSED_DIR / "convfinqa_train.jsonl")          # 可选
FILINGS = str(PROCESSED_DIR / "filings_earnings.jsonl")         # 可选（10-K/纪要）

EVAL_FPB  = str(PROCESSED_DIR / "fpb_eval.jsonl")
EVAL_FIQA = str(PROCESSED_DIR / "fiqa_eval.jsonl")

# 训练集文件清单（存在哪个用哪个）
TRAIN_FILES = [FPB, FIQA, FINQA, TATQA, CONV, FILINGS]
# 采样权重（与你的中间角色匹配：sentiment:news → quant:numeric → dialogue:risk）
TRAIN_WEIGHTS = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0]

def get_experiment():
    cfg = make_train_config(
        model_id=DEFAULT_MODEL_ID,  # "Qwen/Qwen3-8B"
        middle_nodes=4,             # research / fundamental / quant / risk
        epochs=1,
        train_bs=2, eval_bs=2, grad_accum=8,
        phase1_edges_only=True,     # 先只训边，稳定 Dense Communication
        phase2_unfreeze_at_epoch=None,
        dtype="bf16",
        out_dir=str(Path(PROCESSED_DIR).parents[1] / "outputs" / "exp_realdata_dense"),
        log_every=10, eval_every=400, save_every=1200,
    )
    # 评估：用 FPB / FiQA 验证情绪线；数值题建议你另开脚本做 Em/数值误差
    eval_file = EVAL_FIQA if Path(EVAL_FIQA).exists() else (EVAL_FPB if Path(EVAL_FPB).exists() else None)
    return cfg, TRAIN_FILES, TRAIN_WEIGHTS, eval_file