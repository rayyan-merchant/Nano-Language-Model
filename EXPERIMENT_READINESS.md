# Experiment readiness audit

**Decision: NO-GO for the 5,000-step pilot and the 42-run study.**

The integrated implementation passes real-data CPU and Colab T4 development
smokes. Neither independent Kaggle environment has been qualified. Passing
unit tests or a smoke on a different dependency profile is insufficient for GO.
No pilot, 5,000-step training, 10,000-step training or full study was launched.
The locked baseline, E01–E14 definitions and seeds 42/123/456 are preserved.

## Actual execution evidence (2026-10-11)

| Execution | Result | Limit |
|---|---|---|
| M1 preprocessing/BPB tests | 8 passed | Fixtures plus later real acquisition |
| M2 engine tests | 12 passed, 1 CUDA skip | Windows CPU |
| M3 storage/results tests | 3 passed | Local immutable export/restore |
| Final Windows full suite | 31 passed, 1 skipped, 184.73 s | Python 3.10.11, torch 2.5.1+cpu |
| Final Colab full suite | 32 passed, 50.16 s | Python 3.13.15, torch 2.11.0+cu130, Tesla T4 |
| Real Windows integration | Passed | E01 4 updates; E02 2 updates; 512 validation bytes |
| Real T4 float16 integration | Passed | Same short workflow; not pinned research profile |
| Independent Colab data acquisition | Passed | Same raw, tokenizer and manifest hashes |
| Package dependency check | Passed | Windows installed environment |

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
| 9 | Persistent export/recovery | PASS local snapshot/restore and corruption rejection; Colab-to-PC export evidence tracked separately; Kaggle download and mounted Drive qualification PENDING |
| 10 | GPU execution across both Kaggle accounts | PENDING: actual T4 CUDA/AMP execution passed, both Kaggle accounts and pinned GPU dependencies unverified |
| 11 | All-config benchmark procedure | IMPLEMENTED in docs/commands.md; all-14 GPU benchmark NOT EXECUTED |
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

## Reproduction commands actually executed

From the Windows checkout (the prepared real-data directory is ignored):

```text
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m nano_lm.verification --dataset data/processed/tiny-shakespeare-v3 --output outputs/logs/m4-final-cpu --device cpu
.venv\Scripts\python -m pip check
```

On the actual Colab T4, the final process ran `pytest.main(['-q'])`, prepared
Tiny Shakespeare into a fresh `independent-gpu` directory, asserted the BPE hash,
then called `verify(root, data, output, device='cuda', precision='float16')` and
`export_run` on run directories. Source/modules/configs and actual prepared data
were transferred explicitly; this was not a clean committed research checkout.
Process ID: `54f5b121e98b45cfa59ff63d8463128f`.

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
