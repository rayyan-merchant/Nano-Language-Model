# Nano Language Model

One from-scratch GPT research pipeline for Tiny Shakespeare. Research status is
**NO-GO** until the gates in EXPERIMENT_READINESS.md pass. Short CPU/Colab GPU
checks are evidence of implementation correctness, not pilot/ablation results.
No pretrained weights or black-box Transformer architecture are used.

## Install

Locked research profile: Python 3.10/3.11, torch 2.5.1, numpy 1.26.4,
tokenizers 0.21.0, pytest 8.3.4, matplotlib 3.9.4. Use an isolated environment.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python -m pytest -q
```

For CPU install torch from `https://download.pytorch.org/whl/cpu` first.
Linux/Kaggle/Colab use the same commands with the selected environment's `python`.
Do not install the CPU wheel into a GPU environment. Modern Colab Python 3.13
cannot install this locked profile directly: provision a compatible isolated
Python or explicitly qualify a new dependency lock before research. See
`docs/platforms.md`. The newer Colab profile tested here is smoke-only evidence.

## Prepare and verify real data

```text
python -m nano_lm.prepare --output data/processed/tiny-shakespeare
python -m nano_lm.prepare --output data/processed/tiny-shakespeare --validate
python -m nano_lm.verification --dataset data/processed/tiny-shakespeare --output outputs/logs/cpu_audit --device cpu
```

Preparation uses an immutable original Karpathy Tiny Shakespeare revision and
expected SHA256. If local HTTPS trust fails, `--transport git` fetches the same
commit without disabling TLS. New preparation/audit directories must not exist.
No source text is normalized. 90/5/5 splits precede vocabulary fitting and windows.
Character and byte BPE artifacts, split statistics and hashes are validated.

## Short training, resume and inference

These examples are **2-step smoke checks**, with a smaller effective token
budget and validation byte cap, not approved research settings:

```text
python -m nano_lm.train --dataset data/processed/tiny-shakespeare --experiment E01 --run-kind smoke --steps 2 --tokens-per-update 128 --micro-batch-size 1 --eval-every 1 --checkpoint-every 1 --eval-max-bytes 512
python -m nano_lm.generate --checkpoint RUN/checkpoints/step_000002.pt --prompt "To be" --max-new-tokens 80 --greedy
python -m nano_lm.plot_run RUN/checkpoints/step_000002.pt outputs/figures/smoke_run
```

Replace RUN with the printed unique directory. Resume uses the same exact
configuration and `--resume RUN/checkpoints/step_000002.pt --steps 4` (total target,
not additional steps). Extend an approved 5k run to 10k the same way; constant LR
and optimizer/scheduler/scaler/RNG/data sampling continue without schedule reset.
Use E02 for BPE; E01–E14 load from the unchanged registry. All meaningful training
settings have CLI options. Research requires a clean committed checkout; smoke
and benchmark modes are capped at 100 steps. Long research jobs require separate
user approval; no bulk 42-run launch is built into initialization.

Baseline remains width 192, 6 heads, 4 blocks, context 128, 4x MLP, GELU,
sinusoidal positions, dropout 0.1, AdamW 3e-4, 5,000 steps and seeds 42/123/456.
Main comparisons target 4096 tokens/update. Microbatch size controls accumulation;
invalid divisibility fails explicitly. Generation protocol is fixed in
`configs/generation_protocol.json` before scientific runs.

## Platforms, storage and analysis

Thin `notebooks/kaggle_research.ipynb` and `notebooks/colab_research.ipynb` call
this package. For GPU audit add `--device cuda --precision float16` to verification.
GPU absence fails; float32 is baseline and AMP alternatives need qualification.

At checkpoint boundaries, `--export-root DURABLE_ROOT` publishes verified immutable
snapshots. Cloud research requires this option. The caller must verify that the
path is actually persistent: `/content` and unsaved `/kaggle/working` are ephemeral.
See `docs/storage.md` for export, verification, restoration and session-expiry gates.
Only load trusted project checkpoints; PyTorch optimizer/RNG artifacts use pickle.

`nano_lm.results` validates run schemas, hashes, canonical variants, completed
checkpoints and evaluation policy, merges both members' outputs, rejects smoke
or incomplete research evidence, and produces per-run CSV, mean/sample-SD tables
and genuine-metric figures. Missing seeds stay missing. See `docs/commands.md`.

## Layout and limitations

`src/nano_lm`: data/tokenizers, explicit attention/positions/blocks/GPT, evaluation,
training, checkpointing, generation, storage, collection, figures and audits.
`configs`/`experiments`: locked definitions and inference protocol. `tests`: fast
numerical, integration, resume, storage and compatibility tests. `docs`: protocols,
readiness evidence and limits. `api`/`report`: final delivery milestones still pending.

Validation BPB excludes a common 256-byte conditioning prefix and scores each
suffix token once with up to the configured preceding token context. Prefix and
suffix are tokenized separately to force a shared original-byte boundary.
This is conditional likelihood; BPE/character contexts cover different byte spans.
Token losses/perplexities must not be ranked across tokenizers. Test data is
reserved for final evaluation. See `docs/bpb.md` for the exact policies.

The second member verifies and runs the same code; pipeline ownership stays in
this instance. Work on feature branches and reviewed PRs. Never force-push or merge
without authorization. E01, E02, E03, E04, E06, E08 and E09 seed42 each completed an individually authorized
5000-update research run with verified published checkpoints; their reports are in
`report/pilots`. The broader three-seed study remains separately gated. No deployed
API is claimed. Read EXPERIMENT_READINESS.md before further experiment approval.
