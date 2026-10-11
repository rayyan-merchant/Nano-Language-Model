# Nano Language Model

A from-scratch causal GPT for Tiny Shakespeare, shared by two students. The
initial foundation implements explicit attention, all three positional methods,
strict configuration, character tokenization, synthetic CPU smoke training,
checkpoint resume and own-checkpoint CLI inference. Research training has not run.

## Setup

Python 3.10 or 3.11 is recommended. Create a local environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python -m pytest -q
```

Linux, Colab and Kaggle use `python -m pip install -r requirements.txt` and
`python -m pip install -e . --no-deps` in the selected environment. For CPU-only
PyTorch install `torch==2.5.1` using the official CPU wheel index first:
`python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu`.
The exact validated Windows CPU dependency snapshot is in
`requirements-cpu-windows.lock.txt`; notebook accelerator dependencies must be
recorded separately. Do not replace a notebook's CUDA build with a CPU wheel.

## Run a CPU smoke check

Use a fresh directory each time. This uses synthetic repeated text and a tiny
24-wide model, not E01 or Tiny Shakespeare training.

```text
python -m nano_lm.smoke --run-dir outputs/logs/local_smoke --steps 3
python -m nano_lm.smoke --run-dir outputs/logs/local_smoke --steps 6 --resume outputs/logs/local_smoke/step_000003.pt
python -m nano_lm.generate --checkpoint outputs/logs/local_smoke/step_000006.pt --prompt "To be" --max-new-tokens 20 --greedy
```

Expected behavior: smoke writes JSON metadata, append-only metrics and distinct
step checkpoints; generation prints the prompt, model-generated continuation and
parameters. A smoke checkpoint is not a trained research model. Only load trusted
project checkpoints; optimizer and RNG state use PyTorch pickle serialization.

## Project map

- `src/nano_lm`: tokenizer, partition/batching, attention, positions, blocks/GPT,
  generation, checkpoint management and CPU smoke runner.
- `configs/baseline.json`, `experiments/registry.csv`, `experiments/configs`:
  locked E01–E14 definitions, each with three seeds.
- `tests`: numerical attention, causal influence, shapes, positions, gradients,
  overfit, checkpoint compatibility, exact CPU resume and subprocess CLI tests.
- `docs`: research controls, collaboration, assignment compliance and progress.
- `notebooks`: thin platform entry point; `api`, `report`, `experiments/analysis`:
  documented future deliverables rather than fake implementations.

Baseline: width 192, 6 heads, 4 layers, context 128, 4x MLP, GELU, sinusoidal
positions, dropout 0.1, AdamW 3e-4, 5,000 steps, seeds 42/123/456.
Run allocation remains unassigned. Subsequent pilot/GPU jobs require approval.

## Dataset and limits

Required source: [Hugging Face Tiny Shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare).
No dataset has been downloaded or prepared yet. See `data/README.md`.
BPE training, BPB evaluation, research training/validation, mixed precision,
analysis plots, FastAPI hosting and final checkpoint distribution are pending.
There are no research losses, trained samples or deployed service to report.
Read `docs/progress.md` before using this foundation for experiments.

After bootstrap, fetch `origin/main`, create your own descriptive branch and use
PRs. Never force-push or overwrite another member's work. See `AGENTS.md`.
