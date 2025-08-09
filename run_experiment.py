# run_experiment.py
from pathlib import Path
import argparse
from training.trainer import run_training
from configs.experiments.exp_002_trading_firm import get_experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-data", action="store_true", help="先生成合成交易场景数据")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=300)
    args = parser.parse_args()

    if args.generate_data:
        # 直接调用生成器脚本
        import subprocess, sys
        gen_py = Path(__file__).resolve().parent / "data" / "dataset_generator.py"
        cmd = [sys.executable, str(gen_py), "--n-train", str(args.n-train), "--n-eval", str(args.n-eval)]  # type: ignore
        subprocess.check_call(cmd)

    cfg, train_file, eval_file, field = get_experiment()
    run_training(train_file=train_file, eval_file=eval_file, field=field, cfg=cfg)

if __name__ == "__main__":
    main()