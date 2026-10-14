# Research protocol

## Locked study

E01 is the approved baseline in `configs/baseline.json`. E01â€“E14 in the registry
are the only initial training configurations. Each variant changes one intended
factor. E02 replaces the tokenizer; E13/E14 derive MLP dimensions from width,
and vocabulary size is always resolved from the fitted tokenizer. No interaction
configurations are permitted without explicit approval.

Stage A: correctness, numerical attention, causal invariance and tiny overfit.
Stage B: explicitly approved baseline pilot to assess the 5k budget.
Stage C: 14 configurations x seeds 42,123,456 = 42 approved 5k runs.
Stage D: select 2â€“3 strongest configurations on validation and retain E01;
continue saved model/optimizer/scheduler/scaler/RNG state from 5k to 10k.
Stage E: optional approved prolonged study after reviewing evidence. Do not
presume overfitting or double descent. No substantive training during bootstrap.

## Data and comparison controls

Split original decoded UTF-8 text contiguously at floor(0.90*N) and floor(0.95*N)
Unicode code points, then tokenize partitions independently. No window crosses
boundaries. Record original download byte hash separately from re-encoded text
hash, URL, revision, byte count, boundaries, filtering (none), tokenizer artifact
and hash. Fit both vocabularies only on training text. Fail on unseen characters;
do not invent an unknown-token policy after inspecting held-out results.

Matching seeds, optimizer procedure and 5k stop points across variants. Initial
engineering choice: 4096 target training tokens/update; effective sequence batch
sizes are 64/32/16 at context 64/128/256, with configurable accumulation for memory.
This target is provisional until pilot validation and does not alter the locked
model factors. Record actual counts and adjustments. Equal steps/tokens do not
imply equal FLOPs, source bytes or runtime, especially across BPE/character models.

Constant 3e-4 is the initial documented schedule, independent of stop step. Any
approved schedule replacement must be defined before the study and retain its
horizon/regime on 5k-to-10k resume. Initial precision is float32; mixed precision
requires validation. No test-set selection. Use fixed prompts and generation
seeds; greedy and temperatures 0.7/1.0/1.2 form an inference study, not new training
configurations. Preserve all results, failures and best/final checkpoint lineage.

## BPB gate

Character and BPE losses/perplexities must not be ranked directly. BPB evaluation
is pending implementation: score the same original held-out text, sum token NLL
in nats and divide by log(2) times exactly covered original UTF-8 bytes. Define
common conditioning-prefix exclusion, context truncation, special-token policy,
byte offsets and token-boundary differences explicitly; every scored token must
be counted exactly once. Validate byte coverage and likelihood before E02
comparisons. Track effective original-text coverage of each token context.

## Run and analysis requirements

Unique run IDs include experiment, seed and timestamp/nonce. Store immutable
resolved config, Git commit/dirty state, source/split/tokenizer hashes, hardware,
OS/platform, Python/PyTorch/CUDA/dependency versions and precision. Logs record
steps, token/byte exposure, train/validation loss, perplexity, BPB, generalization
gap, LR, gradient norm, elapsed time, throughput, GPU peak memory and checkpoints.
Append and flush JSONL; write atomic checkpoints; refuse output-directory reuse.
Reconcile an interruption's log/checkpoint discrepancy instead of discarding it.
GPU/AMP resume requires scaler and CUDA RNG. Keep durable copies off ephemeral
Kaggle/Colab runtimes; report inaccessible storage. Respect legitimate account
quotas. No paid tracker is required.

Across all three seeds report mean and sample standard deviation (ddof=1).
Separate hypotheses, observations, interpretations and uncertainty. Include
negative results and hardware/parameter-count/context confounds; no unsupported
significance claims or test-selected favorable checkpoints.

## Unified implementation update

Pipeline ownership stays in this instance; the second member independently
verifies and later executes approved runs from the exact shared commit. Modules
are not split across agents. Implementation policies are in docs/bpb.md,
docs/storage.md and docs/platforms.md. The baseline and 14 definitions remain
unchanged. Fixed inference protocol is configs/generation_protocol.json.

Training tokens count actual target exposure, including AMP-skipped attempts.
training_bytes counts those targets' original source bytes, including repeats;
unique_target_bytes tracks their union. These are target-exposure measures, not
unique corpus epochs, input-memory traffic or FLOPs. Effective bytes per token
can be computed from these exact counts. train_seconds excludes evaluation/IO;
eval_seconds and per-segment wall-clock time are separate recorded outcomes.

Full sliding validation and a fixed common conditioning prefix are canonical.
The blocks evaluator and capped smoke metrics are different policies and must
never be silently substituted in the scientific comparison. BPB still has
context/token-boundary confounds explained in docs/bpb.md. Constant LR is the
unchanged initial regime: extending the stop point never changes its schedule.

## Separately approved final-model deviation — 2026-10-14

The user discontinued the planned three-seed grid for time constraints and
explicitly authorized one new combined F01 seed42 trajectory from scratch to
10,000 updates. Historical E01–E14 definitions, seed42 records and the original
design above remain unchanged. F01 is not a single-factor ablation or evidence
of additive effects: BPE1024, width240, eight blocks, RoPE, six heads, context128,
GELU, MLP4x and dropout0.1; 6,046,560 parameters, untied head. Preserve AdamW,
constant3e-4, weight decay0.01, clipping1.0, FP32, microbatch4/accumulation8 and
4096 targets/update. Forty full sliding validations score the common55,514-byte
suffix after256 conditioning bytes. Forty validation states and the canonical
final immutable summary snapshot are required.

Research is gated on the actual combined CPU/CUDA numerical audit, qualified
T4/freeze, measured resource feasibility, durable smoke publication/independent
retrieval and fresh compatible exact full-state restoration with smoke-only
continuity. Select the minimum validation BPB before one reserved final test
assessment with the same256-byte conditioning exclusion; never select or train
again from test results. No additional architecture, seed or extension is
authorized. Record the changed aggregate hash caused by F01.json separately
from the byte-verified unchanged scientific implementation fingerprint.
See report/final/F01_architecture_decision.md. F01 acceptance is currently pending.
