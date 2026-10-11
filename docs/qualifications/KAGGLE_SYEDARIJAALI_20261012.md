# Independent Kaggle GPU qualification: `syedarijaali` (2026-10-12)

**Status: compute qualification PASS; fresh saved-output recovery INCONCLUSIVE.** All work is engineering qualification evidence, not research measurements. No 5,000-step pilot or 42-run study was started.

## Identity, source, and isolation

- The account owner confirmed that Kaggle username `syedarijaali` is their new account. The official CLI OAuth session was switched after securely backing up the earlier `srijaali` session. No credentials are in this report or repository.
- Git source: clean detached `36ffee3926f12f58582b3502f725aecc529a1f8d` from reviewed and merged main. Core/model/configuration/dependency files are unchanged from Rayyan’s `b2ef1ca842ff6b0794b802f1baae58226732674f` source. Code hash: `cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6`.
- Private hardware probe: <https://www.kaggle.com/code/syedarijaali/nano-gpu-probe-20261012-3768dc8a>. Private qualification output version 1: <https://www.kaggle.com/code/syedarijaali/nano-gpu-qualification-20261012-34acfa25>. The downloaded saved metadata confirms `enable_gpu: true`, `machine_shape: NvidiaTeslaT4`; its image digest is in `evidence/syedarijaali_saved_kernel_metadata.json`.
- Every qualification run has a unique private notebook ID and unique run/checkpoint paths. The local repository and existing Python environments were not changed by Kaggle execution.

## Actual hardware and approved environment

- Native runtime: Python 3.13.15, torch 2.11.0+cu128, CUDA available; two Tesla T4 GPUs with 15,360 MiB shown by `nvidia-smi` on each. The qualification used GPU 0 only, without parallel training.
- Isolated research profile: Python **3.11.14**, torch **2.5.1+cu124**, CUDA **12.4**, NumPy **1.26.4**, tokenizers **0.21.0**, pytest **8.3.4**, matplotlib **3.9.4**. `pip check` and CUDA assertions passed. Native packages were untouched.
- The resolved dependency freeze matches Rayyan’s qualification line for line except for the editable package source commit. His and this account’s GPU inventory both report two T4s with 15,636,037,632 bytes total per device. PyTorch peak allocated memory below excludes reserved memory and driver overhead.

## Tests, data, and short CUDA integration

- Full applicable suite: **33 passed, no skips**, pytest 31.21 seconds, subprocess wall time 33.60 seconds.
- Fresh Tiny Shakespeare acquisition matched every audited source, split, partition, tokenizer, and manifest comparison. Raw text: 1,115,394 bytes; train 1,003,854, validation/test 55,770 each. Character/BPE vocabularies: 65/1024, fit on training only.
- Source SHA256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`; character tokenizer: `ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77`; BPE tokenizer: `2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55`; manifest: `76b594a1979107795734f8d2cda08b56c1520c8d44883dbaa5a61dbee53ba16f`.
- Real-data E01/E02 CUDA float16 smoke passed short training, validation, checkpoint reload, generation, local storage restore, and exact resume. Model, optimizer, scheduler, scaler, RNG, and loss/validation histories matched. This smoke is labeled development evidence.

## All-14 short GPU benchmark and separate overhead

Ten actual updates per model on GPU 0, seed 42, float32, 4,096 target tokens per update. These runs include warmup. The 5,000-step values are linear **training-only projections**, not measured full-run durations. Full sliding validation, checkpoint serialization plus flush/fsync, and verified local export were timed separately. Export timing excludes Kaggle output publication and download.

| ID | Steps/s | Tokens/s | Peak allocated MiB | Projected 5k train min | Full val s | Save/fsync s | Local export s |
|---|---:|---:|---:|---:|---:|---:|---:|
| E01 | 4.58 | 18741 | 133.34 | 18.21 | 18.12 | 0.100 | 0.295 |
| E02 | 5.16 | 21144 | 142.99 | 16.14 | 8.38 | 0.124 | 0.353 |
| E03 | 4.42 | 18089 | 133.46 | 18.87 | 20.93 | 0.102 | 0.293 |
| E04 | 5.25 | 21508 | 133.71 | 15.87 | 18.95 | 0.101 | 0.291 |
| E05 | 5.36 | 21955 | 127.90 | 15.55 | 18.97 | 0.100 | 0.295 |
| E06 | 5.36 | 21972 | 128.34 | 15.53 | 17.52 | 0.101 | 0.288 |
| E07 | 5.40 | 22123 | 140.34 | 15.43 | 20.21 | 0.100 | 0.284 |
| E08 | 3.73 | 15259 | 166.10 | 22.37 | 27.49 | 0.148 | 0.421 |
| E09 | 2.80 | 11462 | 199.37 | 29.78 | 36.30 | 0.182 | 0.611 |
| E10 | 5.39 | 22090 | 116.78 | 15.45 | 15.82 | 0.073 | 0.207 |
| E11 | 2.71 | 11091 | 109.10 | 30.77 | 13.21 | 0.103 | 0.286 |
| E12 | 10.53 | 43144 | 204.16 | 7.91 | 45.35 | 0.103 | 0.290 |
| E13 | 5.26 | 21538 | 115.53 | 15.85 | 15.03 | 0.062 | 0.185 |
| E14 | 5.37 | 21988 | 155.35 | 15.52 | 24.10 | 0.151 | 0.461 |

Full validation used 55,770 validation bytes, excluding 256 conditioning bytes. Character models scored 55,514 tokens/bytes; E02 BPE scored 24,386 tokens over the same 55,514 bytes. Validation, save/fsync, and local export totals across the 14 configurations were 300.38, 1.55, and 4.561 seconds respectively. BPB/token-boundary comparison limitations remain as documented in `docs/bpb.md`; no scientific quality claim is drawn from 10-step losses.

## Warmup-aware representative benchmark

Fifty actual updates per configuration; first ten excluded and 40 measured. Projections remain training-only.

| ID | Steps/s | Tokens/s | Peak allocated MiB | Projected 5k train min |
|---|---:|---:|---:|---:|
| E01 | 5.38 | 22050 | 132.84 | 15.48 |
| E09 | 2.84 | 11628 | 198.37 | 29.35 |
| E12 | 10.49 | 42954 | 202.66 | 7.95 |
| E14 | 5.39 | 22063 | 155.98 | 15.47 |
| E02 | 5.28 | 21643 | 142.49 | 15.77 |

## Saved output and recovery

- Kaggle saved qualification output version 1 includes the complete run/checkpoint metadata and weights. The compact downloaded archive is 20,277,543 bytes, SHA256 `47a93fd1ee804506ced5393b673bc394cc7e17a1cc274a979323f5a5a108f81c`; this matches the saved manifest. The compact archive retains the complete smoke checkpoint and benchmark metadata/logs; full benchmark weights remain in the saved notebook output.
- Local Mac inspection passed archive CRC, path safety, snapshot manifest verification, CPU checkpoint load, and generation from the downloaded artifact. Recovered smoke checkpoint SHA256: `915251ce689f4d8cf3cb1ff9cf01b33da090138f0fa085dca2cf3ec55effb9f1`. Generated smoke text is untrained and has no quality meaning.
- A separate private fresh-runtime recovery was submitted at <https://www.kaggle.com/code/syedarijaali/nano-gpu-recovery-20261012-e3ffc064> with the saved qualification notebook as its sole source and a requested 1,800-second limit. Kaggle listed it as started at 2026-10-11 19:34:03 UTC. At 20:14 UTC (over 40 minutes later), the official CLI still reported `KernelWorkerStatus.RUNNING`, with zero published log bytes and no listed output files. The actual cell, whether the saved output mounted, generation, and exact GPU continuation are **unverified**. The cause of the delayed remote run is unknown; no second recovery was submitted.

## Limits and decision

- Short benchmarks and projections are qualification measurements, not study measurements. Forced quota expiry and publication latency were not tested. Actual study training, the 5,000-step pilot, and GPU model-quality claims remain untested.
- Rayyan’s account passed separately. The earlier `srijaali` account still launched CPU-only runtimes despite GPU requests; this report uses the independently authenticated `syedarijaali` account and does not explain that account-side discrepancy.
- **Account-level recommendation: compute checks PASS, durability/recovery gate INCONCLUSIVE. Study-wide recommendation: NO-GO until the fresh recovery evidence is verified, both-account discrepancies are reviewed, and pilot approval is explicit.**
