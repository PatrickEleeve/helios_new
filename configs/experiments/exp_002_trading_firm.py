# configs/experiments/exp_002_trading_firm.py
from pathlib import Path
from configs.base_config import make_train_config, PROCESSED_DIR, DEFAULT_MODEL_ID

# 数据文件（建议用 dataset_generator 先生成）
TRAIN_FILE = str(PROCESSED_DIR / "trading_scenarios_train.jsonl")
EVAL_FILE  = str(PROCESSED_DIR / "trading_scenarios_eval.jsonl")
FIELD_NAME = "text"   # run_training() 会读取该字段

def get_experiment():
    """
    返回 (cfg, train_file, eval_file, field)
    - cfg: TrainConfig
    """
    cfg = make_train_config(
        model_id=DEFAULT_MODEL_ID,   # "Qwen/Qwen3-8B"；如需指令版换成 "...-Instruct"
        middle_nodes=4,              # 4个中间节点：可映射为 research/fundamental/quant/risk 等
        epochs=1,                    # 先快速验证收敛
        train_bs=2,                  # 24G卡建议小batch + grad_accum
        eval_bs=2,
        grad_accum=8,
        phase1_edges_only=True,      # Phase-1：只训边（Dense Communication 的关键）
        phase2_unfreeze_at_epoch=None,  # 验证稳定后再设为 1/2 解冻顶点
        dtype="bf16",
        out_dir=str(Path(PROCESSED_DIR).parents[1] / "outputs" / "exp_trading_dense"),
        log_every=10,
        eval_every=200,
        save_every=1000,
    )
    return cfg, TRAIN_FILE, EVAL_FILE, FIELD_NAME