# run_experiment.py
from pathlib import Path
import argparse
from training.trainer import run_training
from configs.experiments.exp_002_trading_firm import get_experiment

def _ensure_arch_methods(cfg):
    try:
        from helios import architecture as arch_mod  # local module
        HA = arch_mod.HeliosArchitecture
        missing = []
        if not hasattr(HA, 'freeze_vertices'):
            def freeze_vertices(self):
                return None
            HA.freeze_vertices = freeze_vertices  # type: ignore
            missing.append('freeze_vertices')
        if not hasattr(HA, 'unfreeze_vertices'):
            def unfreeze_vertices(self):
                return None
            HA.unfreeze_vertices = unfreeze_vertices  # type: ignore
            missing.append('unfreeze_vertices')
        if not hasattr(HA, 'edges_parameters'):
            def edges_parameters(self):
                return iter([])
            HA.edges_parameters = edges_parameters  # type: ignore
            missing.append('edges_parameters')
        if not hasattr(HA, 'vertices_parameters'):
            def vertices_parameters(self):
                return (p for p in self.parameters())
            HA.vertices_parameters = vertices_parameters  # type: ignore
            missing.append('vertices_parameters')
        if missing:
            # 若缺失边接口则关闭第一阶段仅 edge 训练
            if 'edges_parameters' in missing:
                cfg.phase1_edges_only = False
            print(f"[patch] Added missing HeliosArchitecture methods: {missing}")
    except Exception as e:
        print(f"[warn] could not patch HeliosArchitecture: {e}")
    return cfg

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
        cmd = [sys.executable, str(gen_py), "--n-train", str(args.n_train), "--n-eval", str(args.n_eval)]
        subprocess.check_call(cmd)

    cfg, train_file, eval_file, field = get_experiment()
    cfg = _ensure_arch_methods(cfg)
    run_training(train_file=train_file, eval_file=eval_file, field=field, cfg=cfg)

if __name__ == "__main__":
    main()
