# E14 seed 42: width-240 controlled research experiment

## Objective, hypothesis and authorized scope

Question: how does model width192→240 change next-character prediction,
optimization, resource cost and generation on Tiny Shakespeare? Before the run,
additional capacity could improve fitting or cost more without a predictive gain;
no outcome was assumed. One E14 seed 42 run was authorized for5000 updates.
Earlier experiments and the collaborator's E12 work were preserved. No additional
research configuration, seed, hybrid or extension was launched.

## Architecture and correctness

The sole intended override is d_model240. Four independent pre-LayerNorm blocks,
six heads,40dimensions/head, context 128, MLP 240→960→240, GELU, sinusoidal
positions and dropout0.1. Vocabulary65; embedding65×240; LMhead240→65;
attention/output240→240; all LayerNorm dimensions240. No RoPE, learned positions,
ReLU or BPE. Parameters **2,808,960**, **1,004,160 more** than E01
(55.638298% increase); each MLP has462,000parameters.

Twenty CPU and real CUDA checks verify sole override/defaults, repeat seed 42
initialization, independent blocks, projections/counts, manual sqrt40 attention,
causal mask/softmax/value aggregation, preLN residual and linear/GELU references,
shifted next-token CE, finite nonzero gradients across all blocks, future-token
invariance, dropout train/eval behavior, sinusoidal-only positions, overlength
rejection and autoregressive context 128 cropping. Development gradient checks
show connectivity, not longitudinal per-layer research gradients.

Width changes capacity, head dimension, MLP width, initialization fan-in/RNG
consumption and positional frequency spacing together. Same seed does not mean
identical initial tensors across widths. No width-only mechanism is established.

## Reproducibility and dataset

- Clean trained commit:`ab1da4e05005413807d20802fff4153d97ebfa10`.
- Scientific hash:`cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6`.
- Run:`E14_seed42_91b80cdf4163420f8767b58010eb4e9a`; configuration hash:`09eebce689f9a3fb71e59ee6402a2dd46ec0a42b29fbe2e83c6fb2e3cb19f5ac`.
- [Research saved version1](https://www.kaggle.com/code/rayyannmerchant/nano-e14-pilot-20261013-82e48728).
- [Fresh compatible zero-update recovery](https://www.kaggle.com/code/rayyannmerchant/nano-e14-recovery-20261013-1bd9f987).
- Qualified Tesla T4GPU0; Python3.11.14, torch2.5.1+cu124, CUDA12.4.
- Qualified NumPy1.26.4 and complete transitive freeze; native Kaggle packages unchanged.
- OS:Linux-6.18.48+-x86_64-with-glibc2.39; GPU usable memory15,636,037,632bytes.

Pinned Tiny Shakespeare revision6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e.
Raw SHA25686c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
Character tokenizer SHA256ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77.
Train/validation/test bytes1,003,854/55,770/55,770; train-only fitting and contiguous
separate windows. All source/split/tokenizer/manifest reference fields matched.
No test likelihood or test selection was performed.

## Training, evaluation and pre-run persistence

Fresh seed 42; AdamW0.0003constant, weight decay0.01, clipping1.0, float32 CUDA,
microbatch 4/accumulation 8/4,096 targets/update/32 sampled windows/update. Exactly 5,000
authoritative updates,20 validations every 250, all validation checkpoint states
preserved alongside periodic 500 cadence and best-loss selection. No AMP, early
stopping, precision/schedule modification or CPU fallback.

Every validation uses the canonical full sliding evaluator with context 128,
batch 32/dropoutoff,256 common conditioning-only bytes and55,514 scored original
suffix bytes, each target once. Local positions reset for cropped contexts.
BPB=sum token NLL/(ln2×covered bytes); for this character tokenizer BPB=loss/ln2,
perplexity=exp(loss). No invented step-zero or capped research measurement.

Before research, a separate E14 one-update smoke and10-update development
benchmark were published in a saved output version, independently retrieved and
CRC/SHA256/snapshot-verified. A fresh qualified CUDA runtime restored all six
training-state groups and performed inference at1→1, zero additional updates.
The research wrapper verified that published receipt from its same source.
Development updates are separate from research results. Measured benchmark,
full-validation, checkpoint size, export and storage planning receipts accompany
this report. A short projection is not a measured5000-step runtime.

## Recorded outcomes and time accounting

| Measurement | Value |
|---|---:|
| First training loss/update1 |4.33858347|
| First validation loss/update250 |2.41996274|
| Final training loss |1.42570889|
| Final100-update training mean |1.41597478|
| Best update |4750|
| Best validation loss/BPB |1.48464646/2.14189208|
| Final validation loss/BPB |1.48683957/2.14505608|
| Final perplexity |4.42309454|
| Training-only seconds |978.158|
| Full-validation seconds |490.769|
| Peak allocated GPU MiB |177.283|
| Predicted tokens/source-byte exposure |20,480,000/20,480,000|
| Unique target bytes |1,003,848|

Training excludes evaluation and IO. Pipeline timing below spans source clone to
archive-complete log events, excluding queue, saved-output publication, network
retrieval and fresh recovery. Allocated memory excludes reserved/driver memory.
Canonical peak telemetry resets once and samples after optimizer updates, before
validation; full-run readings include earlier validation allocations but do not
measure a separate final-validation/postprocessing peak. The short benchmark
reads its peak before its sole final validation, so its memory number cannot
replace the recorded research-run peak. E01/E13/E14 share this measurement policy.
Target bytes count repeated target exposure, not input traffic, unique examples,
epochs or FLOPs. Sampler RNG and target union matched E01 and were independently
replayed; different width/dropout/global RNG consumption need not match.

```json
{
  "training_updates_per_second": 5.111646107083857,
  "tokens_per_training_second": 20937.302454615477,
  "checkpoint_seconds": 3.5161309520001396,
  "export_seconds": 11.913577622001185,
  "journal_seconds": 18.734722878007176,
  "generation_seconds": 186.7449725670001,
  "archive_seconds": 34.08662167600005,
  "clone_to_archive_pipeline_seconds": 1903.250505494
}
```

## All twenty measured width comparisons

| Update | E13 loss | E01 loss | E14 loss | E13 BPB | E01 BPB | E14 BPB |
|---|---:|---:|---:|---:|---:|---:|
|250|2.47578366|2.46425110|2.41996274|3.57180081|3.55516284|3.49126824|
|500|2.37145926|2.26721599|2.14415733|3.42129251|3.27090127|3.09336514|
|750|2.19879030|2.11220971|2.00126620|3.17218387|3.04727447|2.88721683|
|1000|2.07306145|1.99768526|1.89425395|2.99079548|2.88205062|2.73283078|
|1250|2.00823062|1.92482598|1.83786761|2.89726435|2.77693689|2.65148249|
|1500|1.94477494|1.86354672|1.76308051|2.80571716|2.68852962|2.54358750|
|1750|1.88900764|1.81413472|1.72174470|2.72526196|2.61724317|2.48395254|
|2000|1.86540537|1.75925914|1.68743306|2.69121108|2.53807444|2.43445131|
|2250|1.82072929|1.72154866|1.64597196|2.62675712|2.48366971|2.37463558|
|2500|1.80133328|1.69282241|1.62271385|2.59877459|2.44222650|2.34108123|
|2750|1.76986721|1.66527225|1.60459348|2.55337865|2.40248002|2.31493905|
|3000|1.74559284|1.63945360|1.58088056|2.51835814|2.36523157|2.28072855|
|3250|1.72042366|1.61751165|1.56445143|2.48204668|2.33357603|2.25702632|
|3500|1.69722521|1.60199293|1.53952033|2.44857839|2.31118726|2.22105834|
|3750|1.68016693|1.58823999|1.52954764|2.42396849|2.29134596|2.20667080|
|4000|1.66561163|1.57893347|1.52319082|2.40296964|2.27791948|2.19749984|
|4250|1.65082152|1.56335215|1.51420351|2.38163202|2.25544039|2.18453389|
|4500|1.63188757|1.54364829|1.49790343|2.35431611|2.22701373|2.16101785|
|4750|1.62459682|1.53466801|1.48464646|2.34379777|2.21405793|2.14189208|
|5000|1.61153841|1.53601210|1.48683957|2.32495848|2.21599704|2.14505608|

## Capacity, performance and measured resources

| Run | Width/head/MLP | Parameters | Best BPB | Final BPB | Best update | Training s | Validation s | GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
|E13|144/24/576|1,021,824|2.32495848|2.32495848|5000|941.393|310.326|132.291|
|E01|192/32/768|1,804,800|2.21405793|2.21599704|4750|916.708|420.908|151.712|
|E14|240/40/960|2,808,960|2.14189208|2.14505608|4750|978.158|490.769|177.283|

E14 minus E01 (negative likelihood/BPB delta favors E14):

| Metric | Absolute delta | Relative % |
|---|---:|---:|
|best_loss|-0.05002155|-3.25944|
|final_loss|-0.04917253|-3.20131|
|best_bpb|-0.07216585|-3.25944|
|final_bpb|-0.07094096|-3.20131|
|final_training_loss|-0.04294454|-2.92408|
|training_seconds|61.45008091|6.70334|
|validation_seconds|69.86149391|16.59781|
|peak_gpu_bytes|26812416.00000000|16.85446|
|parameters|1004160.00000000|55.63830|

## Learning dynamics: observations and interpretation boundaries

Observed3k→5k validation loss change (improvement positive):
0.09404099. Recorded3k–5k fitted slope:
-0.04761045nats/1000updates.
Best checkpoint is4750; final minus best loss is
0.00219312.
These scheduled observations do not establish an asymptotic plateau, significant
overfitting or a general width effect. Same-step generalization gaps compare a
dropout-on sampled training batch with dropout-off full validation and are noisy.
Training averages all positions of sampled windows, including short conditioning
near each window start; sliding validation scores each target with up to128prior
characters. Thus loss gaps also reflect these conditioning differences.
No per-layer gradient time series or FLOP measurement was recorded.

Aggregate actual pre-clipping gradient statistics:

```json
[
  {
    "experiment": "E01",
    "median": 0.4978237450122833,
    "p95": 0.5419636189937591,
    "maximum": 1.8328568935394287,
    "clipped_updates": 7,
    "clipping_fraction": 0.0014,
    "first100_mean": 0.4534429593384266,
    "last100_mean": 0.5208895719051361,
    "scope": "Aggregate pre-clipping research norms; no longitudinal per-layer gradient data"
  },
  {
    "experiment": "E14",
    "median": 0.497776061296463,
    "p95": 0.5397634744644165,
    "maximum": 2.0595805644989014,
    "clipped_updates": 4,
    "clipping_fraction": 0.0008,
    "first100_mean": 0.4677902448177338,
    "last100_mean": 0.5095063006877899,
    "scope": "Aggregate pre-clipping research norms; no longitudinal per-layer gradient data"
  }
]
```

## Complete genuine generations

All 100 best/final protocol slots are preserved byte-for-byte in generation.json
and generation.md: five fixed prompts, greedy(seed 42), temperatures0.7/1.0/1.2
with seeds42/123/456,256 new characters. Best/final identical checkpoint:
False; duplicate slots are explicitly the same weights when true.
No output was rewritten. Five predetermined best greedy examples follow; all
sampled failures remain in the complete grid. Repetition fractions describe
repeated8-character windows and are not coherence or significance scores.

### Prompt 'To be, or not to be'

Checkpointstep_004750.pt; seed 42; greedy;256new tokens.

```text
To be, or not to be the season of the world,
And the seas of the prince of the prince,
And the profess of the prince of the prince
That the profess of the prince of the court.

SICINIUS:
What should be the season of the world of the world
That the profess of the prince of th
```

### Prompt 'ROMEO:\n'

Checkpointstep_004750.pt; seed 42; greedy;256new tokens.

```text
ROMEO:
I will not the grace of the courts of the world,
And the season of the prince of the prince
That thou art the profess of the prince of the world
That the profess of the prince of the court.

SICINIUS:
What should be so much a brother to the prince,
And the
```

### Prompt 'JULIET:\n'

Checkpointstep_004750.pt; seed 42; greedy;256new tokens.

```text
JULIET:
And the man of the courts of the courts,
And the seas of the prince of the country
And the prince of the prince of the prince of the world
That the profess of the prince of the court.

SICINIUS:
What should be so much a brother to the prince,
And the profe
```

### Prompt 'First Citizen:\n'

Checkpointstep_004750.pt; seed 42; greedy;256new tokens.

```text
First Citizen:
And the world of the words of the words of the words
And the seas of the prince of the prince,
And the profess of the prince of the prince
That the profess of the prince of the court.

SICINIUS:
What should be the season of the world of the world
That the 
```

### Prompt 'The king'

Checkpointstep_004750.pt; seed 42; greedy;256new tokens.

```text
The king of the courts of the courts,
And the prince of the prince of the prince
That thou art the prince of the prince of the world
That the profess of the prince of the court.

SICINIUS:
What should be so much a brother to the prince,
And the profess of the prin
```

Aggregate fixed-protocol repetition measurements:

```json
[
  {
    "experiment": "E01",
    "checkpoint_policy": "best",
    "greedy": true,
    "temperature": 1.0,
    "samples": 5,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.5228915662650603,
    "replacement_characters": 0
  },
  {
    "experiment": "E01",
    "checkpoint_policy": "best",
    "greedy": false,
    "temperature": 0.7,
    "samples": 15,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.0013386880856760394,
    "replacement_characters": 0
  },
  {
    "experiment": "E01",
    "checkpoint_policy": "best",
    "greedy": false,
    "temperature": 1.0,
    "samples": 15,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.0,
    "replacement_characters": 0
  },
  {
    "experiment": "E01",
    "checkpoint_policy": "best",
    "greedy": false,
    "temperature": 1.2,
    "samples": 15,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.0013386880856760394,
    "replacement_characters": 0
  },
  {
    "experiment": "E01",
    "checkpoint_policy": "final",
    "greedy": true,
    "temperature": 1.0,
    "samples": 5,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.5437751004016065,
    "replacement_characters": 0
  },
  {
    "experiment": "E01",
    "checkpoint_policy": "final",
    "greedy": false,
    "temperature": 0.7,
    "samples": 15,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.00562248995983936,
    "replacement_characters": 0
  },
  {
    "experiment": "E01",
    "checkpoint_policy": "final",
    "greedy": false,
    "temperature": 1.0,
    "samples": 15,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.0026773761713520714,
    "replacement_characters": 0
  },
  {
    "experiment": "E01",
    "checkpoint_policy": "final",
    "greedy": false,
    "temperature": 1.2,
    "samples": 15,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.0008032128514056325,
    "replacement_characters": 0
  },
  {
    "experiment": "E14",
    "checkpoint_policy": "best",
    "greedy": true,
    "temperature": 1.0,
    "samples": 5,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.38714859437751004,
    "replacement_characters": 0
  },
  {
    "experiment": "E14",
    "checkpoint_policy": "best",
    "greedy": false,
    "temperature": 0.7,
    "samples": 15,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.0024096385542168677,
    "replacement_characters": 0
  },
  {
    "experiment": "E14",
    "checkpoint_policy": "best",
    "greedy": false,
    "temperature": 1.0,
    "samples": 15,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.0013386880856760394,
    "replacement_characters": 0
  },
  {
    "experiment": "E14",
    "checkpoint_policy": "best",
    "greedy": false,
    "temperature": 1.2,
    "samples": 15,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.0,
    "replacement_characters": 0
  },
  {
    "experiment": "E14",
    "checkpoint_policy": "final",
    "greedy": true,
    "temperature": 1.0,
    "samples": 5,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.4465863453815261,
    "replacement_characters": 0
  },
  {
    "experiment": "E14",
    "checkpoint_policy": "final",
    "greedy": false,
    "temperature": 0.7,
    "samples": 15,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.002677376171352086,
    "replacement_characters": 0
  },
  {
    "experiment": "E14",
    "checkpoint_policy": "final",
    "greedy": false,
    "temperature": 1.0,
    "samples": 15,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.0008032128514056251,
    "replacement_characters": 0
  },
  {
    "experiment": "E14",
    "checkpoint_policy": "final",
    "greedy": false,
    "temperature": 1.2,
    "samples": 15,
    "mean_continuation_characters": 256,
    "mean_repeated_8gram_position_fraction": 0.0002677376171352108,
    "replacement_characters": 0
  }
]
```

Training phrase overlap is checked on ten predetermined outputs; ordinary phrases repeat and longest exact overlap is an indicator, not proof of memorization. See checkpoint_diagnostics.json. No test text or external model was used to select or improve outputs.

## Durable artifacts and independent acceptance

Archive size623,713,697bytes.
SHA256:`e41393c4a97c2ec565558df4ca0997d74275cf62e69c57b466c4794855b26fd7`.
All62exact archive entries passed independent CRC/path-safety/
SHA256 checks. All 20 checkpoint counters/history/optimizer/scheduler states and
21 immutable snapshot manifests were verified. Native snapshots remain in saved
outputs; the compact archive holds every distinct resumable checkpoint and logs.
Best/final CPU inference passed. Fresh qualified CUDA restoration preserved exact
model, optimizer, scheduler, scaler, Python/NumPy/torchCPU/CUDA RNG and sampling/
history state at5000→5000, zero additional updates. CPU/CUDA recovery generation
matched. Receipt files, hashes and retrieval locations accompany this report.
The detailed report/comparisons are published as lightweight Git evidence;
large checkpoints, full journals and datasets remain outside Git.

## Failures, tests, limitations and next action

The authoritative5000-step journal records no research failure/resume/replay.
Development and retrieval/API/queue failures, if any, belong to the operational
record rather than altered research history. Test commands and CI evidence are
in evidence/final_checks.json and operations.md; CUDA success refers to actual
executed cloud receipts, not CPU tests.

One seed gives no variance/SD or statistical significance. Width, capacity, head
dimension, MLP width and initialization change together. Equal target counts do
not imply equal FLOPs/runtime; separate-session T4 timings include noise. No test
selection, additional seeds or unapproved10k extension was performed. Review the
actual quality/cost tradeoff and generation failures, then approve matched seeds
before claiming robust architecture superiority. Any10k continuation requires
separate approval and exact complete saved state; it has not been started.

See interpretation.md for evidence-based qualitative observations and budget
recommendation; operations.md for commands and platform incidents, and
artifact_retrieval.md for immutable saved versions/checkpoint locations.

## Genuine figures

![combined_loss](figures/combined_loss.png)

![generalization_gap](figures/generalization_gap.png)

![gradient_norm](figures/gradient_norm.png)

![learning_rate](figures/learning_rate.png)

![throughput](figures/throughput.png)

![training_loss](figures/training_loss.png)

![training_time](figures/training_time.png)

![validation_bpb](figures/validation_bpb.png)

![validation_loss](figures/validation_loss.png)

![comparison_bpb](comparison/comparison_bpb.png)

![comparison_bpb_training_time](comparison/comparison_bpb_training_time.png)

![comparison_loss](comparison/comparison_loss.png)

![comparison_resources](comparison/comparison_resources.png)

![comparison_throughput](comparison/comparison_throughput.png)

![comparison_training](comparison/comparison_training.png)

![width_bpb_parameters](width_comparison/width_bpb_parameters.png)

![width_bpb_training_time](width_comparison/width_bpb_training_time.png)

![width_resources](width_comparison/width_resources.png)

![width_validation_bpb](width_comparison/width_validation_bpb.png)

![width_validation_loss](width_comparison/width_validation_loss.png)
