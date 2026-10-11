# Rayyan Kaggle GPU qualification

Status: PASS for this account's engineering qualification; study-wide NO-GO. All work is development/
qualification evidence. No 5,000-step pilot or 42-run study is authorized here.

## Source and identity

Fetched origin, fast-forwarded local main and verified a clean working tree at
`b2ef1ca842ff6b0794b802f1baae58226732674f`, the merge of PR #1. Qualification
records are prepared separately on `qualification/kaggle-rayyan-20261011`.
The required six documents were read. Canonical architecture/configurations
remain unchanged. Main contains stale historical wording about draft PR #1;
GitHub live metadata confirms it was merged on 2026-10-11.

Kaggle CLI 2.2.4 lists the authenticated account's notebooks as
`rayyannmerchant`. Its existing OAuth session refreshed normally. No other
member's credentials, account switching or quota workaround was used.
GitHub identity is separately `rayyan-merchant`.

## Actual untouched native runtime probe

Private notebook version 1 completed:
https://www.kaggle.com/code/rayyannmerchant/nano-gpu-probe-20261011-3def63c7

The probe cloned the exact source commit, asserted cleanliness and installed
nothing. Downloaded profile evidence reports:

| Component | Observed |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| torch CUDA build | 12.8 |
| NVIDIA driver | 580.178.04; driver-advertised CUDA 13.0 |
| GPU | Two Tesla T4, compute capability 7.5 |
| Per-device PyTorch total bytes | 15,636,037,632 |
| Per-device free bytes at probe | 15,526,068,224 |
| NumPy / tokenizers | 2.1.3 / 0.23.1 |
| pytest / matplotlib | 8.4.2 / 3.10.0 |

These default versions do not satisfy the approved profile. They were not used
as a silent research-policy replacement. The native runtime remains untouched.

## Approved-profile execution: PASS

The private saved qualification notebook, version 1, completed:
https://www.kaggle.com/code/rayyannmerchant/nano-gpu-qualification-20261011-547032e2

The private recovery notebook, version 1, completed in a fresh GPU runtime:
https://www.kaggle.com/code/rayyannmerchant/nano-gpu-recovery-20261011-d1b2dc68

Source in both runtimes: clean detached `b2ef1ca842ff6b0794b802f1baae58226732674f`.
Code hash: `cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6`.
Docker image digest is preserved in evidence/rayyan_saved_kernel_metadata.json.

uv 0.12.7 provisioned isolated Python **3.11.14**, torch **2.5.1+cu124**,
CUDA **12.4**, NumPy **1.26.4**, tokenizers **0.21.0**, pytest **8.3.4**,
matplotlib **3.9.4**; pip check and CUDA assertions passed in both sessions.
Native Python/PyTorch were untouched. This satisfies the existing research
policy; no alternative profile was adopted. Full resolved freeze is committed.
Recommended shared profile: reproduce this isolated profile and verify the freeze
on the second account before any study approval. The builder installs approved
primary pins but resolves transitive dependencies at execution time: compare its
actual freeze, and review discrepancies rather than silently accepting drift.

Full applicable suite: **33 passed, no skips, 32.60 seconds**; subprocess wall
35.12 seconds. Real E01 four-update and E02 two-update CUDA float16 integration
passed acquisition, training, validation, save/reload, generation and exact resume.
The source's shared nano_lm pipeline is used throughout; orchestration contains
no replacement training logic or architectures. All benchmarks below use GPU 0,
float32, seed 42, 4096 target tokens/update, microbatch 4. Allocated GPU 1 was not
used for parallel training. AMP smoke does not change study precision policy.

## Dataset and tokenizer comparisons: PASS

All source, split, partition, tokenizer and manifest comparisons are true against
the audited repository reference. Dataset: 1,115,394 bytes, train 1,003,854,
validation/test 55,770 each; character/BPE vocabularies 65/1024.

| Artifact | Matched SHA256 |
|---|---|
| Raw text | `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |
| Character | `ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77` |
| BPE | `2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55` |
| Manifest | `76b594a1979107795734f8d2cda08b56c1520c8d44883dbaa5a61dbee53ba16f` |

Independent fresh recovery acquisition reproduced the dataset. Full sliding
validation used the same 55,770 original bytes with 256 conditioning bytes
excluded: 55,514 scored bytes for all models. BPE scored 24,386 tokens; character
55,514. Conditional BPB and token-boundary limitations remain as documented in
docs/bpb.md. No study quality comparison is inferred from short-run losses.

## All-14 actual GPU benchmark and separate overhead

Ten updates per model, warmup included. GPU memory is PyTorch peak allocated
memory, not total driver allocation or both devices' memory. Training-only 5k
figures are linear projections, never measured full-run durations. Checkpoint IO
measures a genuine new torch.save + flush/fsync copy; it excludes the canonical
atomic-link step. Export measures verified local /kaggle/working snapshot IO,
not cloud publication. Full validation is measured separately, float32 sliding.

| ID | Steps/s | Tokens/s | Peak MiB | Projected 5k train min | Full val s | Save/fsync s | Local export s |
|---|---:|---:|---:|---:|---:|---:|---:|
| E01 | 4.59 | 18792 | 133.34 | 18.16 | 18.05 | 0.102 | 0.294 |
| E02 | 5.25 | 21504 | 142.99 | 15.87 | 8.38 | 0.123 | 0.366 |
| E03 | 4.39 | 17989 | 133.46 | 18.97 | 20.99 | 0.102 | 0.296 |
| E04 | 5.34 | 21880 | 133.71 | 15.60 | 19.00 | 0.102 | 0.294 |
| E05 | 5.35 | 21908 | 127.90 | 15.58 | 18.56 | 0.101 | 0.293 |
| E06 | 5.36 | 21959 | 128.34 | 15.54 | 17.14 | 0.099 | 0.291 |
| E07 | 5.19 | 21246 | 140.34 | 16.07 | 20.01 | 0.101 | 0.296 |
| E08 | 3.69 | 15134 | 166.10 | 22.55 | 27.32 | 0.147 | 0.421 |
| E09 | 2.79 | 11410 | 199.37 | 29.92 | 35.63 | 0.186 | 0.615 |
| E10 | 5.41 | 22166 | 116.78 | 15.40 | 15.69 | 0.071 | 0.210 |
| E11 | 2.71 | 11099 | 109.10 | 30.75 | 13.01 | 0.099 | 0.292 |
| E12 | 10.45 | 42820 | 204.16 | 7.97 | 44.71 | 0.098 | 0.290 |
| E13 | 5.27 | 21590 | 115.53 | 15.81 | 14.90 | 0.061 | 0.189 |
| E14 | 5.26 | 21559 | 155.35 | 15.83 | 23.66 | 0.148 | 0.460 |

## Warmup-aware representative benchmark

Fifty actual updates/model, first ten excluded; forty measured updates. Runtime
projection excludes validation, IO, notebook setup and network transfer.

| ID | Steps/s | Tokens/s | Peak MiB | Projected 5k train min |
|---|---:|---:|---:|---:|
| E01 | 5.37 | 22008 | 132.84 | 15.51 |
| E09 | 2.83 | 11582 | 198.37 | 29.47 |
| E12 | 10.65 | 43608 | 202.66 | 7.83 |
| E14 | 5.40 | 22128 | 155.98 | 15.43 |
| E02 | 5.31 | 21732 | 142.49 | 15.71 |

Longer context ran faster in this small-model/fixed-token benchmark. A plausible
confound is fewer microbatches/launches: context 64/128/256 requires 16/8/4
accumulation microbatches respectively. This observation does not establish lower
FLOPs or better scientific performance. Full sliding validation cost increases
substantially with context. Projections must be recomputed from the approved pilot.

## Genuine durable storage and recovery: PASS

Kaggle saved qualification output version 1 contains complete run checkpoints,
metadata and logs. It is a private notebook output, not ephemeral working storage.
Downloaded compact archive: **20,277,595 bytes**, SHA256
`9fd5093ee7129eb34a1dd0b46757be340c32f393017e1ee9af623cea3a093f06`.
The compact bundle retains the complete E01 smoke snapshot and all benchmark
metadata/logs, omitting duplicate benchmark weights; full weights remain in the
saved Kaggle output. Archive CRC, hash, path safety and snapshot manifests passed
local verification. A Windows CPU restore loaded the GPU checkpoint and generated.

Fresh Kaggle recovery mounted
`/kaggle/input/notebooks/rayyannmerchant/nano-gpu-qualification-20261011-547032e2/qualification`.
Recovered step-4 checkpoint SHA256:
`3781b8a0ab6c36ab0f50b010913602fd362e1a9e40249dec412531b10941a150`.
Verified snapshot manifest hash:
`72e9439c1ceaa64b7a79d99a8e1766b0479cdc3adda4b343ed74fa30c7d5e2e2`.
Generation succeeded (`To be        `, untrained smoke, not a quality result).
GPU float16 continuation from 4 to 6 updates exactly matched fresh uninterrupted
six-update model, optimizer, scheduler, scaler, RNG and loss/validation histories.
This tests normal restoration across sessions; forced quota expiration mid-write
and actual AMP overflow were not induced. Unpublished work can still be lost.
Both notebooks completed, releasing their execution runtimes.

A second genuine 20,277,595-byte saved-version download into a new directory
took **82.905 seconds**, including CLI/API/auth/network overhead; its SHA256
matched again. This is retrieval latency, not upload/publication time. Kaggle
output-version publication latency was not separately timed.

Each submission has a new private notebook ID; all runs/snapshots have exclusive
UUID paths. No previous result was overwritten. Qualification records are excluded
from study comparisons by run-kind labels. No 5,000-step pilot or 42-run study ran.

## Commands actually executed

```text
git fetch origin
git switch main
git merge --ff-only origin/main
git status --porcelain
git rev-parse HEAD
git switch -c qualification/kaggle-rayyan-20261011
kaggle --version
kaggle kernels list --mine --page-size 1
kaggle kernels push -p outputs/logs/kaggle-probe-3def63c7 --accelerator NvidiaTeslaT4 --timeout 600
.venv\Scripts\python scripts/kaggle_qualification/build_notebook.py --output outputs/logs/kaggle-qualification-rayyan-20261011-v2 --account rayyannmerchant --phase qualification
kaggle kernels push -p outputs/logs/kaggle-qualification-rayyan-20261011-v2 --accelerator NvidiaTeslaT4 --timeout 3600
kaggle kernels pull rayyannmerchant/nano-gpu-qualification-20261011-547032e2 -p outputs/logs/kaggle-qualification-rayyan-20261011-v2/saved-source --metadata
kaggle kernels output rayyannmerchant/nano-gpu-qualification-20261011-547032e2/1 -p outputs/logs/kaggle-qualification-rayyan-20261011-v2/download --file-pattern '^(recovery_bundle.zip|bundle_manifest.json|requirements-qualified.txt|bootstrap.log)$' --quiet
.venv\Scripts\python scripts/kaggle_qualification/inspect_bundle.py outputs/logs/kaggle-qualification-rayyan-20261011-v2/download/recovery_bundle.zip outputs/logs/kaggle-qualification-rayyan-20261011-v2/download/bundle_manifest.json outputs/logs/kq-547032e2
.venv\Scripts\python scripts/kaggle_qualification/build_notebook.py --output outputs/logs/kaggle-recovery-rayyan-20261011 --account rayyannmerchant --phase recovery --source-kernel rayyannmerchant/nano-gpu-qualification-20261011-547032e2
kaggle kernels push -p outputs/logs/kaggle-recovery-rayyan-20261011 --accelerator NvidiaTeslaT4 --timeout 1800
kaggle kernels output rayyannmerchant/nano-gpu-recovery-20261011-d1b2dc68/1 -p outputs/logs/kaggle-recovery-rayyan-20261011/download --file-pattern '.*(recovery.json|failure.json|bootstrap.log)$' --quiet
```

Kaggle ran isolated `python -m pytest -q`, pip check, canonical prepare/verify,
`python -m nano_lm.experiments --root /tmp/nano-source --dataset /tmp/qualification-data --benchmark-all --output /kaggle/working/qualification/all14 --device cuda --steps 10`,
and shared train/evaluate/export/restore entry points. Exact subprocess output,
settings, unique IDs and resolved dependency freeze are in evidence JSON files.
For reproducible submission, use the committed builder with a new output path;
its source commit is deliberately fixed to this qualification target. Do not
change it implicitly when qualifying another commit. Saved notebook code contains
this orchestration embedded outside the clean core checkout.

## Failures, limitations and remaining gates

- Initial sandboxed OAuth refresh could not write credentials; the authorized
  normal CLI refresh succeeded. No credential values were printed/committed.
- Pulling notebook source with a /1 suffix returned 403; pulling its base ID worked
  (only one version exists); output download explicitly used /1 and succeeded.
- nbformat missing-cell-ID and platform conversion SyntaxWarnings were nonfatal.
- Native default profile violates policy; isolated approved CUDA profile passed.
- Second member's GPU/driver/dependency details and independent qualification are
  unknown. The two fresh sessions here belong to Rayyan, not two team members.
- Exact cross-GPU numerical identity is not promised. Compare provenance, hashes,
  tests and compatible resume within each account; use the locked seed protocol.
- Fifty-update benchmarks are short; GPU peak allocated memory excludes reserved
  memory/driver overhead. Cloud output-publication latency was not timed.
- Mounted Google Drive and forced expiry were not tested; actual Kaggle saved-output
  publication, download, independent CPU load and fresh GPU recovery were tested.

**Account-level engineering qualification: PASS. Study-wide recommendation: NO-GO.**
Next: second member independently repeats this exact source/profile qualification
and reviews discrepancies. Only then request separate pilot approval. Do not
change methodology, architecture, E01â€“E14, seeds or study precision from this report.

Primary setup references:
[PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/),
[uv Python management](https://docs.astral.sh/uv/guides/install-python/),
[Kaggle saved notebook versions](https://www.kaggle.com/docs/notebooks),
[official Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).
