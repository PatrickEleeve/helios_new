from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

from training.trainer import main as train_main
from configs.experiments.exp_003_realdata import get_experiment


def main():
	parser = argparse.ArgumentParser(
		description="Train Helios on real datasets (FPB/FiQA/FinQA/TatQA/ConvFinQA/Filings)."
	)
	parser.add_argument(
		"--finqa",
		type=str,
		default=None,
		help="Optional: path to raw FinQA json/jsonl to convert and include in training",
	)
	parser.add_argument(
		"--finqa-out",
		type=str,
		default=str(Path(__file__).resolve().parent / "data" / "processed" / "finqa_train.jsonl"),
		help="Where to write the converted FinQA JSONL",
	)
	parser.add_argument("--max-seq-len", type=int, default=None, help="Override max_seq_len")
	parser.add_argument("--research-rounds", type=int, default=None, help="Override research rounds")
	parser.add_argument("--risk-rounds", type=int, default=None, help="Override risk rounds")
	parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
	parser.add_argument(
		"--phase2-unfreeze-at-epoch",
		type=int,
		default=None,
		help="Set the epoch index to unfreeze vertices (Phase-2). E.g., 1 or 2.",
	)
	args = parser.parse_args()

	cfg, train_files, train_weights, eval_file = get_experiment()

	# Prepare FinQA from raw if provided (accept file or directory)
	if args.finqa:
		finqa_py = Path(__file__).resolve().parent / "data" / "finqa_prepare.py"
		if not finqa_py.exists():
			print("[error] data/finqa_prepare.py not found. Update repository or add the converter.")
			return

		finqa_arg = Path(args.finqa)
		finqa_train_in: Path | None = None
		finqa_dev_in: Path | None = None

		if finqa_arg.is_dir():
			# Common layouts: <root>/dataset/train.json or <root>/train.json(.jsonl)
			cands_train = [
				finqa_arg / "dataset" / "train.json",
				finqa_arg / "dataset" / "train.jsonl",
				finqa_arg / "train.json",
				finqa_arg / "train.jsonl",
			]
			cands_dev = [
				finqa_arg / "dataset" / "dev.json",
				finqa_arg / "dataset" / "dev.jsonl",
				finqa_arg / "dev.json",
				finqa_arg / "dev.jsonl",
			]
			finqa_train_in = next((p for p in cands_train if p.exists()), None)
			finqa_dev_in = next((p for p in cands_dev if p.exists()), None)
		else:
			# A file was passed; if it exists, use it. Else try typical dataset path next to it
			if finqa_arg.exists():
				finqa_train_in = finqa_arg
			else:
				base = finqa_arg
				cands = [
					base.parent / "dataset" / base.name,
					base.parent / "dataset" / "train.json",
					base.parent / "dataset" / "train.jsonl",
				]
				finqa_train_in = next((p for p in cands if p.exists()), None)
				# dev alongside
				devc = [
					base.parent / "dataset" / "dev.json",
					base.parent / "dataset" / "dev.jsonl",
				]
				finqa_dev_in = next((p for p in devc if p.exists()), None)

		if not finqa_train_in:
			print(f"[error] Could not locate FinQA train file from: {args.finqa}\n"
				  f"        Tried typical paths like dataset/train.json. Pass a valid file or directory.")
			return

		# Convert train
		train_out = Path(args.finqa_out)
		cmd = [sys.executable, str(finqa_py), "--in", str(finqa_train_in), "--out", str(train_out)]
		print(f"[prep] converting FinQA train: {' '.join(cmd)}")
		subprocess.check_call(cmd)
		train_files.append(str(train_out))

		# Convert dev/eval if available
		if finqa_dev_in:
			eval_out = train_out.parent / "finqa_eval.jsonl"
			cmd_dev = [sys.executable, str(finqa_py), "--in", str(finqa_dev_in), "--out", str(eval_out)]
			print(f"[prep] converting FinQA dev: {' '.join(cmd_dev)}")
			subprocess.check_call(cmd_dev)
			# Prefer FinQA dev as eval if preset eval missing
			if not eval_file or not Path(eval_file).exists():
				eval_file = str(eval_out)

	# Optional overrides to reduce memory / change schedule
	if args.max_seq_len is not None:
		cfg.max_seq_len = int(args.max_seq_len)
	if args.research_rounds is not None:
		cfg.research_rounds = int(args.research_rounds)
	if args.risk_rounds is not None:
		cfg.risk_rounds = int(args.risk_rounds)
	if args.epochs is not None:
		cfg.epochs = int(args.epochs)
	if args.phase2_unfreeze_at_epoch is not None:
		cfg.phase2_unfreeze_at_epoch = int(args.phase2_unfreeze_at_epoch)

	# Filter to existing files
	existing = [p for p in train_files if Path(p).exists()]
	if not existing:
		print(
			"[error] No training files found. Use --finqa to convert your raw FinQA, or generate synthetic data via\n"
			"        python data/dataset_generator.py --n-train 2000 --n-eval 300"
		)
		return

	val_files = [eval_file] if (eval_file and Path(eval_file).exists()) else []

	print("[realdata] training files:")
	for p in existing:
		print(f"  - {p}")
	if val_files:
		print("[realdata] eval file:")
		for p in val_files:
			print(f"  - {p}")

	# Single aggregated run across all files
	train_main(train_cfg=cfg, helios_cfg=None, train_files=existing, val_files=val_files)


if __name__ == "__main__":
	main()

