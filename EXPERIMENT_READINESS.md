# Experiment readiness audit

## Scoped E02 seed42 Colab pilot — 2026-10-12

The second member explicitly authorized E02 seed42 for exactly 5,000 optimizer
updates after source, CUDA/software and durable-storage qualification. This one-run
exception completed on Colab Tesla T4 at clean main
`2cf39518492652e2ed05e459318e6166e4275814`, with the identical E01 canonical core
hash and exact Python3.11.14 / torch2.5.1+cu124 / CUDA12.4 dependency freeze.
35 automated tests passed. Drive smoke export/restore and CUDA continuation passed
before research. Twenty full sliding validations, all twenty checkpoint counters,
21 durable snapshots, fresh-runtime best/final inference and the complete genuine
generation grid were verified. The archive was independently retrieved in checked
parts, reconstructed, CRC/file-hash audited, and loaded for CPU inference.

Best/final update is 5,000; conditional validation BPB is 2.02659521. Measured
comparison and parameter/context/exposure confounds are in the
[E02 report](report/pilots/E02_seed42/E02_seed42_analysis.md). A postprocessing Drive
hard-link manifest error was repaired with staged rename; no extra training
updates occurred. Definitions in the configuration registry remain unchanged.

**Study-wide NO-GO remains. No further research seed, configuration or 10k
continuation is authorized.** Historical E01 and qualification records follow.

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


## E04 seed42 complete — 2026-10-12

Individually authorized E04 learned-position run completed exactly5000 updates
and all20 full sliding validations on rayyannmerchant's qualified T4 GPU0.
Clean trained source b260264521ceb280f356b82ecd8cf4b69cb8af21; unchanged audited
scientific core cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6.
Parameters1829376 (+24576). Best/final5000 loss1.55111692,BPB2.23778869;
all matched E01/E03 validation losses are lower than E04. Extra capacity and
canonical initialization RNG consumption limit pure representation attribution.
Training1001.78457s,validation413.79723s,pipeline1766.15717s,peak152.0874MiB.
Original Kaggle5e7b13ef/v1 and fresh recoveryd31e262c/v1 are COMPLETE.
Independently retrieved407744239-byte archive SHA256
60b25eeab43606cedf1ef1e4ab86f9dfbbe10ccc366c1bac290015996e3a86fe:
58 files verified;20 checkpoints and21 snapshots preserved. Independent CPU
inference and exact fresh CUDA full-state roundtrip/generation passed,zero
extra optimizer updates. Read-only monitoring connections dropped/reconnected;
research journal has no failures/resumes. D recovery prevented C-space problems.
Windows38 passed/1 CUDA skip; CPU/CUDA13 learned-position checks passed; training
commit CI green before submission. Actual20 checkpoint counters and character
BPB identity independently audited.100 unedited generation rows/50 duplicate
best-final pairs; strong greedy loops/malformed words retained.14 measured figures,
three-way character comparisons and separate contextual E02 BPB summary:
report/pilots/E04_seed42/E04_seed42_analysis.md plus interpretation.md.
E01/E02/E03 evidence and canonical definitions unchanged. E04 seed42 alone is
complete; broader three-seed study remains separately gated. No other experiment,
seed or10k extension launched. PR8 is reviewable and is not merged by this task.

## E06 seed42 completed and independently recovered -- 2026-10-12

The explicitly authorized E06 four-head run completed5000 optimizer updates and
all20 full sliding validations on TeslaT4/Python3.11.14/torch2.5.1+cu124,
float32,seed42. Clean trained source fc08577afb63d4bd3ecad43bf33355c65ddd985c;
scientific hash cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6.
Only head count6->4 differs; head dimension48,parameters1804800. All13 CPU
and actualCUDA attention checks passed. Final Windows suite41 passed/1 CUDA
skip (285.62s); no full Kagglepytest suite is claimed. Best4750 loss1.52865285
BPB2.20537989; final5000 loss1.52920435 BPB2.20617553. Training917.232s,
validation389.111s, allocated peak145.712MiB. Twenty resumable states and21
snapshots are published in private Kaggle notebook version1
rayyannmerchant/nano-e06-pilot-20261012-62c30c02. Independent402500855-byte
archive SHA256 bf224ad20d4eca03359d08da820289ac1e9cef2c8f082f9d94e229f8edb29530
passed58 file hashes/ZIP checks and CPU inference. Fresh savedCUDA recovery
rayyannmerchant/nano-e06-recovery-20261012-cab9c7aa verified exact full state
and best/final inference with zero extra updates. No research failure/resume
event.100 genuine samples and13 figures retained, including greedy loops.
Small E01 advantage is descriptive for one seed, not significant evidence.
Full report: report/pilots/E06_seed42/E06_seed42_analysis.md and interpretation.md.
Large artifacts are onD and saved Kaggle output, excluded fromGit. PR9 is for
review; no merge or extra configuration/seed/extension was performed. Historical
E01-E04 evidence preserved; E05 Colab execution untouched. Broader three-seed
study remains separately gated/NO-GO, not authorized by this scoped completion.

## E08 seed42 completed and independently recovered -- 2026-10-13

The explicitly authorized six-block E08 run completed exactly5000 optimizer
updates and20 full sliding validations on TeslaT4 GPU0,Python3.11.14,
torch2.5.1+cu124/CUDA12.4/NumPy1.26.4,float32. Clean trained source
72fdd4b64610e8e6df178e7fd452ec9429faab8f; scientific hash
cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6.
Only n_layers4->6 differs;2694528 parameters (+889728 capacity confound).
CPU/CUDA13 depth checks passed. Before launch Windows44 passed/1 CUDA skip;
actual full Kaggle suite45 passed33.84s, including CUDA AMP-resume tests.
Best4750 loss1.48713993/BPB2.14548940; final5000 loss1.48858993/BPB2.14758132.
Training1511.357s, validation593.279s, clone-to-archive pipeline2602.147s;
allocated peak166.104MiB. Lower best BPB than E01 by3.097% costs64.868% more
training time. One seed does not establish significance or depth-only causality.

Research private saved notebook version1:
rayyannmerchant/nano-e08-pilot-20261013-79fc614d. All20 resumable states and21
immutable snapshots published. Independently downloaded598768138-byte archive
SHA256 f1071463a41c18954cb8eb177b9e8a24ba9a692077ddb97e438c8af49d881fb4
passed59 file hashes/ZIP checks and best/final CPU inference. Fresh saved CUDA
recovery rayyannmerchant/nano-e08-recovery-20261013-75b12285 verified exact model,
optimizer/scheduler/scaler/RNG/sampling state at5000, zero additional updates.
All20 checkpoint counters and character BPB independently audited. No research
failure/resume event.100 unedited generations and15 measured PNG figures retained;
greedy loops/malformed words remain despite improved validation likelihood.
Full report and interpretation: report/pilots/E08_seed42/E08_seed42_analysis.md.
Large artifacts remain onD and saved Kaggle outputs, excluded fromGit.

E06 PR9 was already owner-merged before source synchronization; no formal review
submission is claimed. E08 PR12 is for review and not merged by this task.
Earlier reports/configurations and E07 Colab work are unchanged. No other
configuration, seed or10k extension was launched. E09 depth-framework metrics
remain null/not-run. This scoped E08 run is complete; the broader three-seed
study remains separately gated/NO-GO. Final test evidence is in the E08 report.

## E09 seed42 eight-block research run — completed 2026-10-13

One explicitly authorized E09 seed42 run completed exactly 5,000 optimizer
updates and 20 full sliding validations on rayyannmerchant Kaggle Tesla T4 GPU0.
Fresh initialization; only n_layers4->8 differs from E01. Parameters3,584,256,
+1,779,456 (98.596%) versus E01; depth, capacity and computation change together.
Clean trained source c555269ad6de293c016d88beec2e1437e4289e3f; audited scientific
hash cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6 unchanged.
Python3.11.14/torch2.5.1+cu124/CUDA12.4/NumPy1.26.4 and qualified dependency
freeze matched. Dataset, partitions and character tokenizer matched reference.

CPU and CUDA15 depth checks passed, including independent Q/K/V/sqrt32/causal
attention references and gradients across all eight blocks. Final Windows suite:
46 passed,1 CUDA-only skip122.72s. Actual Kaggle CUDA suite47 passed32.43s.
Real-data CUDA smoke save/load/export/restore/ZIP/Agg/generation passed before
full reseeding. All20 checkpoint optimizer/scheduler counters, journals, BPB
and E01/E08/E09 configuration/environment/sampling/coverage comparability passed.

Best4750 loss1.45571123/BPB2.10014737; final5000 loss1.46725990/BPB2.11680859.
Training1811.126s, full validation739.049s, clone-to-archive pipeline3117.872s;
peak allocated199.370MiB. Best BPB lower5.145% versus E01 and2.113% versus E08,
at97.568%/19.834% extra training time. Single seed; no significance/depth-only
causality. Late improvements diminish, with a final uptick, not sustained
observed deterioration. Greedy loops and malformed language remain.

Private research saved version1: rayyannmerchant/nano-e09-pilot-20261013-b9ea5610.
All20 resumable checkpoints and21 immutable snapshots published. Independently
retrieved794894378-byte archive SHA256
293aaecba3bd794fcd41a58dff08791a63d897bf20a5b913a75b2adbaa3a7b21 passed ZIP CRC,
exact entry set and59 file hashes. Best/final CPU inference passed. Fresh CUDA
saved version1 rayyannmerchant/nano-e09-recovery-20261013-66f79ad5 restored exact
model/optimizer/scheduler/scaler/RNG/sampling/history at5000 with zero extra updates;
its checkpoint hashes and64-token samples match independent CPU recovery.
No research failure/resume event. Local monitoring reconnects/OAuth401 were
resolved through legitimate existing-account refresh, without restarting training.

Complete100-sample fixed grid,18 measured PNG figures,9 vector PDF figures,
all20 matched comparisons, machine-readable results, exact commands and detailed
interpretation are in report/pilots/E09_seed42/E09_seed42_analysis.md.
Large artifacts remain onD and published Kaggle outputs, excluded fromGit.
Historical E08 placeholder E09 values remain untouched; the current three-way
comparison uses the actual recovered E09 measurements.

E08 PR12 was already owner-merged before preparation; this task did not merge it
or claim a formal human review. Main was refetched and remained
 d1ca09df3d20478aabcdd3b4999f8cef6e0d5782. E09 PR13 is for review, no automatic
merge. Earlier reports/configurations and E07's separate work are unchanged.
This scoped run is complete. The broader three-seed study remains separately
gated/NO-GO; no other experiment, seed or10k extension was started. Recommendation:
retain5k screening and consider separately approved selected10k extensions and
additional seeds;5k does not prove convergence.

## E10 seed42 2× MLP research run — completed 2026-10-13

One authorized E10 seed42 run completed exactly5,000 optimizer updates and20
full sliding validations on rayyannmerchant Kaggle Tesla T4 GPU0. Only
mlp_expansion4→2 differs:192→384→192 in all four blocks,1,213,440 parameters,
591,360 fewer than E01 (32.766%). Architecture and capacity change together.
Clean trained source75bc0c4257f0be6a8c0280ddee1dcee52fecb0af; scientific hash
cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6 unchanged.
Qualified Python3.11.14/torch2.5.1+cu124/CUDA12.4/NumPy1.26.4/freeze and canonical
source/tokenizer/partition hashes matched. No test likelihood or prior weights.

CPU/CUDA14 MLP numerical checks passed. Actual Kaggle source suite49 passed31.20s;
final Windows reporting-corrected suite49 passed/1 CUDA-only skip127.39s. The
additional E10 reporting regression passed2 tests0.16s. Real-data CUDA
save/load/export/restore/ZIP/Agg/generation preflight passed before reseeding.
All20 optimizer/scheduler counters,5,000 journal steps,BPB identity and E01/E10
config/profile/data/tokenizer/private-sampling/coverage comparability passed.

Best=final5,000: loss1.57144692/BPB2.26711868/perplexity4.81360807.
Training921.331s, validation315.430s, clone-to-archive pipeline1586.809s;
allocated peak139.705MiB. Best BPB is2.397% above E01. Training time+0.504%,
full validation time−25.060%, memory−7.915%. Fewer parameters do not establish
better quality or faster training. One seed has undefined SD/no significance.
Validation continued improving to the endpoint; severe greedy loops remain.

Private research saved version1: rayyannmerchant/nano-e10-pilot-20261013-403eb78d.
All20 resumable states and21 snapshots published. Independently retrieved
272568662-byte archive SHA256
f20b2394f1d0a99f132caa90a3148cb08a03fb7682b4507c4dce206015d9bf8b passed ZIP CRC,
exact entry set and59 file hashes. Best/final CPU inference passed. Fresh CUDA
rayyannmerchant/nano-e10-recovery-20261013-e2fb429a/1 restored exact model,
optimizer/scheduler/scaler/RNG/sampling/history at5,000 with zero extra updates.
Recovered checkpoint hash b4f0e17427158fb45920c6f2b34a2c270b84a16ac0f58096af68b9e3294a9819
and64-token samples agree across independent CPU and CUDA recovery.

Resolved engineering failures: missing D test parent; default C test-temp ACL;
reporting-only E10 allowlist omission; two interrupted CLI downloads. Failed
attempts were preserved. Authorized output byte ranges plus a retained45MiB
streaming prefix recovered the exact archive. An optional CPU fallback push was
rejected by notebook-title validation before execution, used no compute and was
not retried after local recovery succeeded. No research failure/resume event,
scientific correction or altered trained output. All events are documented.

Report/pilots/E10_seed42/E10_seed42_analysis.md and interpretation.md contain all
20 matched comparisons,15 actual PNG/vector-PDF pairs,100 unedited generations,
resource tradeoffs, hashes and commands. Best/final generation policies duplicate
one weight state, not independent draws. Large evidence stays onD and Kaggle.
E09 PR13 was already merged before preparation; E07 PR11 and earlier records
remain untouched. E10 PR14 is for review and not auto-merged. Main was refetched
at a6b28d7d97f9739dddd24ec4c7c3765bc65265dc. The broader three-seed study remains
gated/NO-GO. No other configuration, seed, hybrid or10k extension was launched.
Keep5k screening; additional seeds and selected extensions require separate
approval. Retain4× for observed predictive quality;2× is a compact candidate with
an explicit likelihood penalty, not an established overall improvement.

## E11 seed42 — measured run and independent archive verified

Sourcee7f77205de0c7eabddf6c299029ef5d4b685ffc8; corehashcdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6 unchanged. Context64/microbatch4/accumulation16/4096targets,1804800parameters. Exactly5000updates/20fullslidingvalidations; best=final5000loss1.49301304/BPB2.15396251. All20resumable states and21native verified snapshots published. Independent402283332-byte archive SHA256a23789511ce10aa70ac3484ca279f2327eee5a9d1d4db319e4ab2f47df7484ce:CRC/all60files/hash/CPUinference/all20counter/private sampler audits passed. All100uneditedgenerations and measuredfigures/report exist. Fresh CUDA full-state restoration remainsPENDING:existing recovery notebookRUNNING/logHTTP500/no receipt. FullE11acceptanceINCOMPLETE; study-wideNO-GO remains. NoE12,otherseed or extension. See report/pilots/E11_seed42/E11_seed42_analysis.md and operations.md.

## E11 recovery gate resolved — 2026-10-13

The existing rayyannmerchant/nano-e11-recovery-20261013-e7000753/1 job isCOMPLETE. Independently retrieved receipt verifies exact model/optimizer/scheduler/scaler/RNG/sampling/training-state restoration,5000→5000zeroadditionalupdates, matching qualified source/profile and best/final CPU/CUDA generation equality. FullE11single-runacceptance isCOMPLETE. The prior queued/HTTP500 delay is retained as historical evidence. No duplicate research/extra seeds/E12/extension. Study-wide approval remains separately gated. See report/pilots/E11_seed42/evidence/cuda_recovery.json.

## E13 seed42 width144 — accepted 2026-10-13

E11exactfreshCUDAacceptance and PR15integration were reverified beforeE13. Clean
trainedsourcea62cd6cebda71d5fbe1606add9efd9f71171076a;scientific hashcdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6unchanged. Onlywidth144/head24/MLP576 mechanicallyderived;1021824parameters(−43.383%). Exactly5000updates/20fullvalidations/context128/microbatch4/accumulation8/4096targets. Best=final5000loss1.61153841/BPB2.32495848,5.009%higherBPBthanE01-best. Training941.393s,validation310.326s,pipeline1625.390s,peak132.291MiB. No recordedresearchfailure/resume;no replay.

Published230564978-byte archiveSHA2560ddce4c76829e961ff862ab61f7d67f9a8ca679260d4fc3a8219fc617becf7e7independently passedCRC/exactinventory/all61hashes/CPUbestfinalinference/all20counters/samplerreplay/21snapshotmanifest parity. FreshqualifiedCUDAallstate5000to5000zeroextra andCPU/CUDAgeneration match passed. Full100uneditedgrid/15PNG/15PDFfigures/machinecomparisons/detailedreport preserved outsideGitforlargefiles. CPU63passed1skip;actualGPU64passed. NoE12/E14/otherseed/hybrid/extension. E13accepted;study-wideapprovalremains separatelygated. See report/pilots/E13_seed42/E13_seed42_analysis.md.


## E14 width240 seed42 — accepted 2026-10-14

E13 PR16 was verified merged before preparation. Frozen trained source
`ab1da4e05005413807d20802fff4153d97ebfa10`; scientific hash
`cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6` unchanged.
Only width240, deriving head40/MLP960; 2,808,960 parameters (+55.638%).
A separate published E14 development artifact passed independent 81-file/CRC/
snapshot/CPU/fresh-CUDA full-state recovery before research. The one authorized
run completed exactly 5,000 authoritative/known physical updates and 20 full
sliding validations, context128/microbatch4/accumulation8/4,096 targets.
Best4750 loss1.48464646/BPB2.14189208; final loss1.48683957/BPB2.14505608.
Best BPB is3.259% lower than E01; training978.158s (+6.703%), validation490.769s,
recorded allocator peak177.283MiB. Source-clone to archive1903.251s.

Research archive623,713,697bytes, SHA256
`e41393c4a97c2ec565558df4ca0997d74275cf62e69c57b466c4794855b26fd7` independently
passed CRC/safe paths/exact62-file inventory/allhashes, all20optimizer/scheduler
counters, sampler replay and best/final CPU inference. All21published snapshot
manifests matched; fresh compatible CUDA final full-state5000→5000zeroextra and
best/final inference matched. All100 unedited generations, 20PNG/20PDF figures,
full E13/E01/E14 comparisons and detailed report are preserved. No research
failure/resume/replay; log-service disconnects retained separately. CPU69passed
1CUDA-only skip; actual CUDA70passed. Single-seed/confounding/peak-sampling scopes
are explicit. No other research run or extension launched; study-wide approval
remains separately gated. See report/pilots/E14_seed42/E14_seed42_analysis.md.
