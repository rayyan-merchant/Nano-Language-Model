# B01 authorization and frozen preparation — 2026-10-14

Reviewed main d43cfbf7e3af6c982a8bcf1509a5f45b8e17ef4b includes merged F01 PR19.
One fresh B01 seed42 trajectory is authorized, maximum5,000 updates, with the
user-defined stopping rule. No test likelihood, other dataset download, other
configuration/seed, tuning, repeated research initialization or extension.

BPE1024/d192/layers6/heads6/head32/RoPE/context128/MLP768/GELU/preLN/dropout.1,
untied LM head:3,062,784 parameters. Independent component count: embeddings
196,608, six blocks2,669,184, finalLN384, head196,608. Train AdamW3e-4constant,
wd.01/clip1/FP32/micro4/accumulation8/4,096 targets per update. Canonical raw/BPE
hashes and contiguous partitions remain unchanged; no vocabulary refitting.

## Frozen stopping semantics

Initialize reference and counter0 at first full validation250. Later observations
reset reference and counter only when `bpb < reference - 0.001` using unrounded
Python double precision. Equality with that computed threshold does not reset;
next representable values on either side are tested. Small cumulative improvements
can cross the frozen reference threshold later. Absolute minimum BPB is tracked
independently without tolerance; ties retain the first minimum. Counter increments
before eligibility; no artificial reset at2,000. At eligible scheduled boundaries
stop when counter>=6. At5,000, an eligible early-stop condition takes precedence
if simultaneous, otherwise reason budget_exhausted. No observation/update follows
the decision. Stop history and configuration hashes are preserved.

## Planned segmentation and equivalence gate

The existing canonical CLI supports target-total-step exact resume. B01 invokes
it at250,500,... with the same run and last committed checkpoint. Each invocation
uses the original optimizer, constant scheduler, scaler, RNG and sampling state.
These are planned segments, not new models, failures or replay. The original
engine exports at validation and after each segment summary:2*N snapshots for N
observed validations, plus a final stopping-summary snapshot. Preserve all of them.

Development compares two4-update B01 smoke trajectories, uninterrupted versus
2+2 resumed, with the same settings and real Tiny Shakespeare. Model/optimizer/
scheduler/scaler/allRNG and every nontelemetry nested training field—including
loss, validation, gradient norms, source-byte union and private sampling state—
must match exactly. Only measured train/eval times, throughput and peak-memory
telemetry are excluded from trajectory equality: they cannot be identical wall
clock measurements. Their raw inequality is reported, not concealed. Complete
six-field checkpoint save/restore roundtrip requires exact raw equality, including
all telemetry/history. CUDA gate uses uncapped full sliding validation; CPU smoke
may explicitly cap512bytes and is labeled development only.

## Preflight and execution gates

1. Config/count/history parity and independent CPU mathematics; stopping tests.
2. Commit all orchestration and rule before training; no changes to src algorithms.
3. Qualified T4/Python3.11.14/torch2.5.1+cu124/CUDA12.4/NumPy1.26.4/freeze.
4. Actual CUDA math, full suite, segmentation, benchmark/full validation and storage.
5. Publish development snapshot/ZIP, independently retrieve/check CRC/all hashes/CPU.
6. Fresh qualified CUDA exact state and segmented-continuation equivalence.
7. Only then one B01 research trajectory with frozen rule; preserve every validation.
8. Publish/retrieve all real stopping artifacts, CPU and fresh CUDA recovery,
   exact100-slot generation reproduction, inference timing and reports/PR.

Initial read-only quota:21.57GPUh remaining. This is not GPU allocation proof.
At the initial preparation checkpoint, no research run had launched. F01's severe late deterioration motivates B01 but
does not isolate parameter count or clipping as its cause. Early stopping changes
selection opportunities/budget and does not guarantee better text. Original F01
test exposure is acknowledged; it is not used or reevaluated in B01. An external
held-out evaluation may be proposed later with rules frozen before inspection;
none is downloaded or evaluated here.

## Actual CPU preparation evidence

All six-block CPU math/gradient/causal/RoPE-relative checks passed at3,062,784
parameters. Real-data development4-update uninterrupted versus2+2 segmented
state matches all optimization/RNG/nontelemetry-history fields exactly; raw
six-field zero-update snapshot roundtrip also passed. Headless Agg smoke plotting,
generation and raw-byte diagnosis passed. Initial full suite90passed/onefailed/
oneCUDA-onlyskip: the new tokenizer test relied on Windows cp1252 rather than
explicit UTF8. Corrected all new text readers to UTF8; full suite91passed/
oneCUDA-onlyskip in123.63s. No vocabulary, model or optimizer fix was needed.
These are CPU development checks, not CUDA qualification or research results.

## Actual GPU persistence gate and single research submission

Qualified development notebook `rayyannmerchant/nano-b01-development-20261014-9812dbba/1`
completed with 92 tests passing, actual CUDA architecture checks and uninterrupted
versus segmented full-sliding validation equivalence. Its 203,352,277-byte archive
SHA256 is `a2c23046cd5b40864ac38efd2f743d6e2cf9b5ce164f30745331653075d9b1e6`.
Independent download, ZIP CRC, safe exact inventory, 59 file hashes and CPU
inference passed. The first local downloader failed before transfer because an
eager fallback required a missing legacy run_id; the filename resolver was fixed
and tested without rerunning development or changing scientific execution.

Fresh notebook `rayyannmerchant/nano-b01-development-recovery-20261014-44cabfe4/1`
completed: all six full-state fields match on step2-to-step2 roundtrip with zero
updates. Its two development-only resumed updates then match the published
uninterrupted step4 scientific trajectory exactly. This is development evidence,
not additional research training. Receipts are under `report/balanced/evidence/development`.

Only after these gates, the single research notebook
`rayyannmerchant/nano-b01-research-20261014-33317a00/1` was submitted. Its frozen
execution commit is `c2cf1c165d09e0fc8f64a2df714755a4d910d469`; later local reporting
or download-helper commits do not change that execution source. Native Kaggle
log-stream reconnections are read-only transport retries, not training restarts.
Final acceptance requires actual stopped-run publication, independent archive
verification and fresh zero-update CUDA recovery; submission is not acceptance.

## Stopped-run publication and acceptance

One research version1 completed at the first eligible early stop:4,500 updates,
18 full sliding validations, best3,000 BPB1.9849309521722194 and final4,500
BPB2.044457963659289. All18 checkpoints and37 canonical native snapshots are
retained. Actual immutable archive604,825,667bytes has SHA256
`41742da357777a6aee40bd7bc4a4063ee1e3d07f94b94e2344f082252a467edf`.
Independent retrieval/CRC/safe exact96-file inventory/all hashes/all counters/
stopping rule/sampling RNG/byte coverage/CPU inference passed. Fresh qualified
CUDA notebook d2c9b763 version1 matches all six complete training-state fields
at4500→4500, adds zero optimizer updates, verifies37 snapshots and reproduces
the complete100-slot grid; independently compared256-token IDs all match.

See `report/balanced/B01_acceptance.md`, `artifact_retrieval.md`, actual analysis
and model-selection reports for receipts, commands, figures and limitations.
No original test likelihood, additional corpus, model, seed or extension was
executed. Early stopping is normal frozen termination; there is no unmeasured
5,000-step result. Predictive competitiveness is distinct from greedy loops,
invalidUTF8 output and other genuine generation failures.
