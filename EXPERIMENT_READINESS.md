# Experiment readiness audit

## Scoped E01 pilot authorization — 2026-10-12

Rayyan explicitly authorized E01 seed42 for exactly5000 optimizer updates on
qualified Kaggle rayyannmerchant despite the unavailable second GPU account.
This one-account exception permits the pilot only. Study-wide NO-GO remains;
no remaining seeds, E02–E14 or 10k continuation are authorized. Execution and
preservation procedure: docs/pilot_execution.md. The authorized pilot completed
5000 updates and20 full validations. Published output was recovered in a fresh
compatible CUDA environment with exact state restoration and zero extra updates.
Actual results and independent archive verification:
[E01 seed42 report](report/pilots/E01_seed42/E01_seed42_analysis.md).

**Current decision: scoped E01 seed42 pilot executed; NO-GO for the distributed
42-run study. No other seed/configuration or 10k extension is authorized.**

The integrated implementation passes real-data CPU and Colab T4 development
smokes. Rayyan's Kaggle environment is now qualified at merged main b2ef1ca; the second
member's independent account remains unqualified. Passing
unit tests or a smoke on a different dependency profile is insufficient for GO.
The historical audit below preceded the explicitly authorized 2026-10-12 pilot.
No10k extension, other research seed/configuration or full study was launched.
The locked baseline, E01â€“E14 definitions and seeds 42/123/456 are preserved.

## Actual execution evidence (2026-10-11)

| Execution | Result | Limit |
|---|---|---|
| M1 preprocessing/BPB tests | 8 passed | Fixtures plus later real acquisition |
| M2 engine tests | 12 passed, 1 CUDA skip | Windows CPU |
| M3 storage/results tests | 3 passed | Local immutable export/restore |
| Windows full suite before CLI fix | 31 passed, 1 skipped, 184.73 s | Python 3.10.11, torch 2.5.1+cpu |
| Latest Windows full suite after CLI fix | 32 passed, 1 skipped, 227.06 s | Forced-CP1252 Unicode CLI regression included; CPU-only environment |
| Final Colab full suite | 32 passed, 50.16 s | Python 3.13.15, torch 2.11.0+cu130, Tesla T4 |
| Real Windows integration | Passed | E01 4 updates; E02 2 updates; 512 validation bytes |
| Real T4 float16 integration | Passed | Same short workflow; not pinned research profile |
| Independent Colab data acquisition | Passed | Same raw, tokenizer and manifest hashes |
| Package dependency check | Passed | Windows installed environment |
| Clean Ubuntu checkout CI | 31 passed, 1 CUDA skip, 20.24 s | Dependency/package installation succeeded; run 38158274754 |
| Colab evidence exported to PC | Passed | 276,461,015-byte archive, SHA256 and 3 snapshot manifests verified; runtime released |

Real integration prepares/validates data, trains both tokenizers, evaluates,
saves/reloads, generates, and compares uninterrupted E01 with 2 updates followed
by exported/restored resumption to 4. Model, optimizer, scheduler, scaler, RNG and
committed loss/validation histories compare exactly within each environment.
CPU and GPU numerical training results are not asserted bitwise equivalent.
Evidence snapshots are development smokes; they must not enter research tables.

## Acceptance checklist

| # | Criterion | Evidence / status |
|---|---|---|
| 1 | All 14 instantiate with only intended factors | PASS: parameter/shape audit below; original JSON definitions unchanged |
| 2 | Real Tiny Shakespeare CPU end to end | PASS: final verification JSON and own-model samples |
| 3 | Partition/window/tokenizer leakage | PASS: split before fitting/tokenization; training samples only train; overlapping windows within train allowed and counted, never across partition boundaries |
| 4 | Independent conditional BPB and byte coverage | PASS: analytic uniform distribution and independent causal-forward reference tests; same original scored suffix; limitations below |
| 5 | Accumulation, shifted targets, mask, dropout, aggregation | PASS: complete automated suite; summed NLL weighted by actual scored tokens/bytes |
| 6 | Full resume incl. sampling/RNG/AMP | PASS in CPU and T4 float16 smokes; same code/hardware resume gate; forced actual overflow/preemption remains untested |
| 7 | Unique IDs and metadata lineage, no silent overwrite | PASS: UUID runs, exclusive atomic artifacts, immutable metadata, journal/history hash, dataset/tokenizer/config/code/Git linkage |
| 8 | Both-member result aggregation | PASS on controlled test outputs: schema/metadata/hash/completeness checks, competing-run rejection and duplicate-copy deduplication; genuine study inputs not yet available |
| 9 | Persistent export/recovery | PASS local/Colab plus genuine Rayyan Kaggle saved output, checksum download, Windows load and fresh Kaggle exact AMP resume; mounted Drive untested |
| 10 | GPU execution across both Kaggle accounts | PARTIAL: Rayyan T4, isolated approved Python 3.11.14 / torch 2.5.1+cu124 passed; second member account PENDING |
| 11 | All-config benchmark procedure | PASS on Rayyan T4: actual all-14 ten-step and five fifty-step warmup-aware benchmarks, separate full-validation/IO timings; second account PENDING |
| 12 | Tables/figures from real recorded metrics | IMPLEMENTED and collector tested; mean/sample SD and completeness explicit; no invented study figures |
| 13 | Fixed generation settings | PASS: configs/generation_protocol.json established before research execution |
| 14 | Extend 5k to 10k without schedule reset | PASS structural/short extension tests: constant LambdaLR independent of stop target and full optimizer restore; no actual 5k/10k run |
| 15 | Full tests and honest failure record | PASS: actual results above; resolved issues and remaining limits below |
| 16 | Final readiness decision | NO-GO until two-environment and scientific/operational qualification gates pass |

## Dataset and tokenizer validation

Pinned Karpathy char-rnn revision: `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`.
Raw UTF-8 SHA256:
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
The file contains 1,115,394 bytes, all ASCII. Contiguous original-text partitions
are train 1,003,854 bytes, validation 55,770, test 55,770; concatenation exactly
reconstructs the source. Tokenizer fitting consumes train only. BPE's fixed byte
alphabet handles unseen bytes without learning held-out merges.

| Artifact | Actual vocabulary / token count | SHA256 |
|---|---|---|
| Character | 65; train/val/test 1003854/55770/55770 | `ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77` |
| Byte BPE | 1024; train/val/test 411158/24516/24905 | `2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55` |
| Dataset manifest | Exact partitions, provenance, hashes, token counts | `76b594a1979107795734f8d2cda08b56c1520c8d44883dbaa5a61dbee53ba16f` |

Both tokenizers round-trip every partition with exact source-byte conservation.
Original text can naturally repeat across partitions; separation prevents window
or fitting leakage, not coincidental literary phrase repetition.

## Actual model audit

Shapes are batch-one forward outputs, followed by per-head attention shape.
Character vocabulary is 65; BPE vocabulary is 1024. All use the existing explicit
from-scratch Transformer. E02 vocabulary/embedding/output size differences are
the intended consequences of tokenizer choice.

| ID | Parameters | Logits shape | Attention shape |
|---|---:|---|---|
| E01 | 1,804,800 | 1,128,65 | 1,6,128,128 |
| E02 | 2,173,056 | 1,128,1024 | 1,6,128,128 |
| E03 | 1,804,800 | 1,128,65 | 1,6,128,128 |
| E04 | 1,829,376 | 1,128,65 | 1,6,128,128 |
| E05 | 1,804,800 | 1,128,65 | 1,6,128,128 |
| E06 | 1,804,800 | 1,128,65 | 1,4,128,128 |
| E07 | 1,804,800 | 1,128,65 | 1,8,128,128 |
| E08 | 2,694,528 | 1,128,65 | 1,6,128,128 |
| E09 | 3,584,256 | 1,128,65 | 1,6,128,128 |
| E10 | 1,213,440 | 1,128,65 | 1,6,128,128 |
| E11 | 1,804,800 | 1,64,65 | 1,6,64,64 |
| E12 | 1,804,800 | 1,256,65 | 1,6,256,256 |
| E13 | 1,021,824 | 1,128,65 | 1,6,128,128 |
| E14 | 2,808,960 | 1,128,65 | 1,6,128,128 |

## Scientific interpretation and risks

BPB is `sum(scored token NLL) / (ln(2) * original scored suffix bytes)`.
The same first 256 source bytes condition character and BPE evaluation and are
excluded from the denominator. Prefix and suffix are separately tokenized at a
common byte boundary; this changes unconstrained BPE segmentation at that boundary.
Sliding evaluation scores every suffix token once with up to L earlier tokens,
reset local positional indices, and no future tokens. Token context lengths imply
different source-byte contexts across tokenizers. These are conditional scores,
not whole-file compression rates. See docs/bpb.md for precise treatment.

Training uses 4096 predicted tokens/update by default across context variants;
microbatch and exact accumulation control memory. Fixed steps and token counts
do not equal equal FLOPs, equal source-byte exposure or equal unique text coverage.
Target byte exposure includes repetitions; distinct target bytes are separately
counted. Training-only throughput/projections exclude validation, checkpoint and
export overhead. Full sliding validation cost and model-specific peak memory need
measurement before approving the pilot. Precision remains an explicitly controlled
setting; CUDA AMP smoke does not authorize changing the float32 baseline protocol.

Snapshots preserve latest and best checkpoints, metadata and journals; older
snapshot directories preserve earlier milestones. Export completion requires a
verified manifest. An expired runtime can lose anything since the last durable
completed export. No software can guarantee survival of unexported /content data.
Restore to a new directory after expiration; never remove an active writer's lock.

## Resolved failures and remaining blockers

- Local Hugging Face endpoint TLS failed; pinned original GitHub source succeeded
  with exact hash verification. No certificate checks were disabled.
- BPE preprocessing repeatedly queried vocabulary size per token; caching removed
  that performance fault. Failed partial preparation directories are retained.
- Windows default test encoding and overly long staging filenames were corrected.
- Accumulation checks now compare gradients/outputs with stated floating-point
  tolerance, avoiding an invalid near-zero Adam parameter equality criterion.
- A test run overlapped a source edit and correctly rejected changed code lineage;
  the final suite and integration were rerun with stable source.
- Colab Python 3.13 / torch 2.11 differs from the pinned 3.10/3.11 / torch 2.5.1
  profile. Qualify the locked GPU profile or obtain approval for a shared new lock.
- Both independent Kaggle accounts, actual Kaggle persistence, mounted Drive,
  full-validation runtime, all-14 GPU benchmarking and genuine multi-seed results
  remain pending. Existing fixture/table tests do not establish real study quality.
- Connector nested-directory export hit a Windows long-path temporary-name
  failure. An archive export recovered full GPU evidence, SHA256
  `9966c4bbc31d3479797ebace698c98d949200460d553d20c69586ad5877d4b03`.
  Three snapshot manifests verified locally, a GPU checkpoint loaded for CPU
  generation, and genuine smoke loss/throughput figures regenerated. The runtime
  was released only after verified archive publication. The failed partial
  transfer directory is retained. Use short archive paths with this connector
  on Windows; Kaggle persistence and mounted Drive remain unverified.

## Reproduction commands actually executed

From the Windows checkout (the prepared real-data directory is ignored):

```text
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m nano_lm.verification --dataset data/processed/tiny-shakespeare-v3 --output outputs/logs/m4-final-cpu --device cpu
.venv\Scripts\python -m pip check
.venv\Scripts\python outputs/logs/verify_gpu_export.py
.venv\Scripts\python -m nano_lm.verification --dataset data/processed/tiny-shakespeare-v3 --output outputs/logs/m4-windows-utf8-cpu --device cpu
```

On the actual Colab T4, the final process ran `pytest.main(['-q'])`, prepared
Tiny Shakespeare into a fresh `independent-gpu` directory, asserted the BPE hash,
then called `verify(root, data, output, device='cuda', precision='float16')` and
`export_run` on run directories. Source/modules/configs and actual prepared data
were transferred explicitly; this was not a clean committed research checkout.
Process ID: `54f5b121e98b45cfa59ff63d8463128f`.

Small evidence is committed under docs/evidence; full checkpoints, archive and
smoke figures remain ignored under outputs/logs/m4-final-gpu. CPU/GPU source hash
matches: `5f1a5e15f888f942a9d3a841c4d32771ad0a83d5d24cacf3de76c0f7a7e1f7fe`.
Development CPU snapshots record dirty source state; Colab records no Git checkout.
These are not clean research runs. The implementation commit is
`ffccc1a39c3a55c509534e5d84264653977fa07b`; later documentation-only commits
preserved that audited source until the Windows CLI output fix below. PR #1 was subsequently merged as b2ef1ca; the earlier evidence below predates that merge:
https://github.com/rayyan-merchant/Nano-Language-Model/pull/1.
Clean-install CI passed:
https://github.com/rayyan-merchant/Nano-Language-Model/actions/runs/38158274754.

Post-export real temperature generation exposed a CP1252 console printing error
for generated Unicode. The CLI now explicitly outputs UTF-8; a subprocess test
forces PYTHONIOENCODING=cp1252 and checks Unicode output. Repeating temperature
0.7, seed 42, eight new tokens from the actual downloaded E02 checkpoint passed.
The initial failure is recorded, not suppressed. This output-only change is later
than the T4 audit; GPU training/evaluation/AMP implementations are unchanged.
The T4 suite above therefore refers to the audited source hash, not a claim that
the additional CLI regression test executed on a GPU. Latest Windows evidence
is in docs/evidence/final_windows_tests.json; final CI is checked separately.
The real-data CPU integration was also rerun successfully after the CLI fix;
docs/evidence/latest_cpu_verification.json records this source and exact resume.
Byte-BPE smoke samples can contain replacement
characters from untrained byte tokens; they are not research-quality samples.

For clean-checkout installation, CLI exports/restores, the short benchmark and
subsequent result collection, follow README.md and docs/commands.md. Neither
benchmark nor approved research commands in those docs were executed as a study.

## Required next action

Have the second member review this implementation and BPB policy. Pin one shared
GPU environment, run full tests and real-data CUDA verification independently in
both Kaggle accounts, verify downloaded snapshots after restarting sessions, and
benchmark all 14 models including full-validation/storage overhead. Record exact
commits, hashes, hardware and dependencies. Resolve discrepancies before changing
NO-GO to GO. Then seek separate approval for the 5,000-step E01 pilot; allocation
and the 42-run study follow the measured pilot and another execution approval.

## Independent Rayyan Kaggle qualification (2026-10-11)

[Complete qualification report](docs/qualifications/KAGGLE_RAYYAN_20261011.md)
records genuine private saved notebook versions and committed machine-readable
evidence. Clean merged main `b2ef1ca842ff6b0794b802f1baae58226732674f` ran
33 tests successfully (32.60 s; no skips), real character/BPE CUDA smoke, exact
AMP resume, all audited data/tokenizer hash comparisons, fourteen ten-update
float32 GPU benchmarks and five fifty-update warmup-aware benchmarks. Full
sliding validation and local checkpoint/export overhead were measured for all14.
Saved Kaggle output was checksum-downloaded, loaded on Windows CPU and restored
in a separate Kaggle runtime. Float16 4-to-6 continuation exactly matched an
uninterrupted run's model/optimizer/scheduler/scaler/RNG and loss/validation histories.

Two Tesla T4s were allocated; only GPU0 trained. Untouched native Python3.13 /
torch2.11 was incompatible with policy. Isolated Python3.11.14 / torch2.5.1+cu124 /
CUDA12.4 / NumPy1.26.4 passed without replacing the base CUDA environment. The
resolved freeze is recorded. No canonical architecture/config/seed/precision
policy changed, no pilot or study launched. Both sessions were Rayyan's account;
they do not qualify the second member.

Historical pending Kaggle persistence, all-14 timing and pinned GPU-profile gates
above are now passed for Rayyan only. Second member's hardware/profile, independent
full workflow and discrepancy review remain pending. Forced expiry/AMP overflow
and mounted Drive were not tested. **Global NO-GO remains.** Next: repeat the exact
source/profile on the second account, reconcile evidence, then seek pilot approval.


## E03 seed42 execution evidence — 2026-10-12

The individually authorized E03 RoPE run completed exactly5000 updates and20 full
sliding validations. Clean training source0dd75296a773081f55a7a6d1da111e482562c1c1;
scientific core hash matches E01. Best/final update5000: loss1.43708178,
conditional BPB2.07327076,1804800 parameters. Train1107.721s; validation447.515s;
complete measured notebook pipeline1951.871s; peakallocated151.71MiB.
Published original notebook82f33b8a version1 and fresh recovery33b2c3ba version1
are COMPLETE on rayyannmerchant. Archive402094081bytes SHA256
57feb9d606a1d733ce3ae301b9e401c12b200f32c1d3c3fca4c47c3c6117b110:
58 files independently verified;20 checkpoints/21 snapshots retained. CPU
inference and fresh CUDA full-state restoration/generation passed with zero
additional updates. Local C-drive extraction failure was recovered to unique D
storage without changing E01 evidence. Report/evidence:
report/pilots/E03_seed42/E03_seed42_analysis.md and interpretation.md.
Pre-execution Windows tests36 passed,1 CUDA skip; actual CUDA RoPE audit13 passed.
E03 improves final loss6.44% with20.84% more training time; greedy generation is
more repetitive on4/5 fixed prompts. Single-seed descriptive comparison only.
No other run or extension launched. E02 is the colleague's independent Colab
execution, outside this evidence. The broader three-seed study remains separately
gated; scoped E01/E03 completion is not global GO. Further experiments/extensions
require authorization. PR6 is reviewable and is not merged.
