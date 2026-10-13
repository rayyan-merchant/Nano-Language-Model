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
without authorization. E01, E02, E03, E04, E06, E08, E09 and E10 seed42 each completed an individually authorized
5000-update research run with verified published checkpoints; their reports are in
`report/pilots`. The broader three-seed study remains separately gated. No deployed
API is claimed. Read EXPERIMENT_READINESS.md before further experiment approval.

## Scoped E11 seed42 context64 evidence

[Detailed E11 report](report/pilots/E11_seed42/E11_seed42_analysis.md):5000updates/20fullvalidations, context64/accumulation16/4096targets,1804800parameters. Best=final5000loss1.49301304/BPB2.15396251. Publishedarchive independently verified and best/final CPUinference passed. Fresh zero-update CUDA recovery now passed exact full-state restoration and CPU/CUDA generation equality; full E11 acceptanceCOMPLETE. NoE12,otherseed or extension authorized or launched.

## Scoped E13 seed42 width144 evidence

[Accepted E13 report](report/pilots/E13_seed42/E13_seed42_analysis.md):5000updates/20fullvalidations,1021824parameters,best=final5000loss1.61153841/BPB2.32495848. Publishedarchive/allfilehashes/CPUinference/freshCUDAfullstate5000to5000zeroextra passed. Smallerparametercountdidnot improvepredictivequalityorobservedtrainingspeed;memory/validationcostlower. All100uneditedgenerations and15PNG/15PDFfigurespreserved. E14/otherseed/extensionnotlaunched;study-wideauthorization separatelygated.


## Scoped E14 width240 seed42 evidence

[Accepted E14 report](report/pilots/E14_seed42/E14_seed42_analysis.md): 5,000 updates,
20 full validations, 2,808,960 parameters; best4750 loss1.48464646/BPB2.14189208,
final BPB2.14505608. Published archive/all62filehashes/CPUinference/freshCUDAfinal
full-state5000→5000zeroextra passed. Best BPB is3.259% belowE01 with55.638% more
parameters and6.703% more measured training time. Complete100-slot grid and
20PNG/20PDF figures preserved; greedy repetition remains despite lowerloss.
No other seed/configuration/extension launched; single-seed and resource scopes
are explicit, and broader study approval remains separate.
