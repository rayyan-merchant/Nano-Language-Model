# Research protocol

## Locked study

E01 is the approved baseline in `configs/baseline.json`. E01–E14 in the registry
are the only initial training configurations. Each variant changes one intended
factor. E02 replaces the tokenizer; E13/E14 derive MLP dimensions from width,
and vocabulary size is always resolved from the fitted tokenizer. No interaction
configurations are permitted without explicit approval.

Stage A: correctness, numerical attention, causal invariance and tiny overfit.
Stage B: explicitly approved baseline pilot to assess the 5k budget.
Stage C: 14 configurations x seeds 42,123,456 = 42 approved 5k runs.
Stage D: select 2–3 strongest configurations on validation and retain E01;
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
