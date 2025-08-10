# Repository Guidelines

## Project Structure & Module Organization
- `helios/`: Core model code — adapters, agents, protocols, and `architecture.py` (graph of roles/edges).
- `training/`: Training loop, data collation, metrics (`trainer.py`, `eval_metrics.py`).
- `configs/`: Base config and experiment presets (`configs/experiments/*.py`).
- `data/`: Synthetic dataset tools (`dataset_generator.py`), outputs in `data/processed/`.
- Top‑level scripts: `run_experiment.py`, `evaluate_text.py`, `standalone_edge_sanity.py`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Generate data: `python data/dataset_generator.py --n-train 2000 --n-eval 300`
- Train (preset): `python run_experiment.py --generate-data` (writes under `outputs/` per config). Note: configs resolve files in `configs/experiments/`.
- Quick sanity run (no project imports): `python standalone_edge_sanity.py --steps 80 --rank-eval`
- Evaluate a prompt: `python evaluate_text.py --prompt "... Decision: " --choice-mode rank`

## Coding Style & Naming Conventions
- Python 3.10+. Indentation: 4 spaces. Follow PEP 8 and type‑hint new code.
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE` (see `DEFAULT_MODEL_ID`).
- CLIs use `argparse` with long flags (`--choice-mode`, `--n-train`). Keep side effects under `if __name__ == "__main__":`.
- Keep tensors and modules on matching device/dtype (pattern used in `helios/*`).

## Testing Guidelines
- No test suite yet. Add `pytest` tests under `tests/` named `test_*.py` (e.g., unit test `EdgeAdapter` shape handling and causal masks).
- Example: `pytest -q` after activating the venv. Prefer fast, CPU‑only unit tests; mock large models.

## Commit & Pull Request Guidelines
- History is minimal; adopt Conventional Commits (e.g., `feat: add risk gate`, `fix: dtype mismatch in EdgeAdapter`).
- PRs should include: purpose, linked issues, how to run (commands), and expected output logs/metrics. Add screenshots or loss curves when training changes behavior.
- Keep diffs focused; document any API changes in docstrings and update example commands in this file if relevant.

## Security & Configuration Tips
- Large models: `DEFAULT_MODEL_ID` may point to local paths; switch to a HF ID (e.g., `Qwen/Qwen3-8B`) if desired. Ensure GPU with `device_map="auto"` and correct `--dtype`.
- Do not commit `outputs/` or `data/processed/` artifacts. Keep secrets and credentials out of the repo and configs.

