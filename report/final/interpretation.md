# F01: observations, interpretation and final-model recommendation

This document accompanies the measured tables, complete unedited generation
grids and acceptance receipts in [the full report](F01_10000_analysis.md).
F01 is one freshly initialized seed-42 combined model. It is neither a
single-factor ablation nor a replicated estimate of an optimized architecture.

## Hypothesis and observed predictive outcome

The pre-run hypothesis was that train-only BPE, RoPE, greater depth and width
could jointly improve next-token prediction and generated text. Interaction
effects were unknown; the individual pilot improvements were not added together.

The observed minimum validation BPB is **1.98376817 at update 1,750**. E02's
recorded best is 2.02659521: a reduction of 0.04282705 BPB, approximately 2.113%
for this one validation suffix and seed. This meets the approved predictive
target. It does not establish better grammar, a causal contribution from each
factor, statistical significance or general superiority.

The model at 5,000 updates scores **2.35443990 BPB**, and the final model at
10,000 scores **3.05346210 BPB**. Those states are worse than E02's best, despite
receiving more training than the selected F01 state. Reporting only the early
minimum would conceal this important negative result. The selected state received
7,168,000 predicted targets; the complete authorized run received 40,960,000.
Their original-byte exposure and actual training cost are recorded separately
in `findings.json`.

The selected checkpoint's accumulated training time is approximately 747.174
seconds, versus the full F01 trajectory's 4,269.206 seconds. This is a
retrospective observation, not a separately executed early-stopping experiment
or proof that an advance 1,750-step budget would generalize to another seed.
The actual experiment paid for all 10,000 updates and 40 validations.
Selection considered 40 F01 observations versus 20 per pilot. More checkpoint
selection opportunities and previous validation-based architecture choices
must be acknowledged; the best-score difference is not an unbiased replicated
estimate of future performance.

## Learning behavior and budget

Validation improves strongly at first, reaches its observed minimum at 1,750,
and shows sustained deterioration over the remainder of the run. Meanwhile,
training loss decreases to 0.92398473; its trailing 100-update average is
0.94353615. This sustained pattern is consistent with overfitting to the training
distribution, rather than a conclusion drawn from one small validation fluctuation.

The last 5,000 updates do not deliver a validation recovery or a measured
double-descent reversal. The report retains every evaluation, including the
intermediate 5,000-step diagnostic. There was no early stop or change to the
learning rate, dropout, precision, optimizer, sampling or scoring policy.

Training loss is measured with dropout on, over sampled training windows;
validation uses dropout-free full sliding conditioning. Thus the numerical gap
combines data-distribution differences with these measurement differences. The
observed curves alone cannot identify a particular layer, attention mechanism or
initialization effect as the cause. Aggregate gradient norms do not demonstrate
vanishing or exploding gradients through individual layers.

**Recommendation:** retain the validation-selected update-1,750 checkpoint as
NanoLM-Final. Retain the final and all intermediate checkpoints as research
evidence, but do not deploy the update-10,000 state merely because it is later.
The authorized task ends at 10,000; no further training, seeds or hyperparameter
searches were launched. This run gives no evidence that extending the same
trajectory would improve held-out prediction.

## Genuine text: repetition, coherence and failures

All five fixed prompts, all three temperatures and all three sampling seeds
remain present for both best and final checkpoints. The full 100-slot grid is
in [generation.md](https://github.com/rayyan-merchant/Nano-Language-Model/blob/5f3bfd8b3a95c346be9f70aa0404e1cdf2fbd430/report/final/generation.md); the predeclared 50-slot 5,000-step grid is in
`step5000_generation.json`. Every output contains 256 new BPE token IDs, not
necessarily 256 characters. Character-model comparisons therefore have unequal
continuation lengths. Automatic string diversity and training-lexicon novelty
are descriptive measures, not grammar or understanding scores.

The best checkpoint's greedy continuations show severe repetitive dialogue.
Across the five prompts, repeated 8-character-gram position fraction averages
0.524674. “I will not be a king” and KING RICHARD III speaker sequences recur;
the ROMEO and JULIET prompts often drift rapidly to those unrelated speakers.
The JULIET continuation includes malformed forms such as “farmery-beding”.
First Citizen also produces repeated “careless” and “sir” sequences. Preserving
the supplied prompt as a literal prefix does not imply semantic responsiveness.

Temperature 0.7 reduces this short-gram repetition average to 0.028552 over
15 fixed best-checkpoint samples, but does not solve coherence. The predetermined
seed-42 outputs contain “pitchrehen”, “guilty-contagine”, “unclawds”, “pimony”,
“corrow” and other malformed or questionable words. Some speaker labels and
sentence fragments resemble Shakespearean dialogue, while relationships between
speakers and statements frequently break down. The original strings are kept
unedited; no external model rewrote them.

Higher-temperature diversity is not sufficient evidence of improved quality.
Fresh compatible CUDA inference exactly reproduced all 100 strings and retained
the generated token IDs. Strict reconstruction of ByteLevel bytes found **five
invalid UTF-8 outputs**, all among the best checkpoint's temperature-1.2 samples.
Native decoding inserted six replacement characters across those strings.
These failures were preserved, not repaired or filtered. Training-text Unicode
roundtrips passing does not guarantee that arbitrary sampled token sequences
form valid UTF-8.

The final checkpoint has less greedy short-gram repetition (mean 0.042478),
yet substantially worse validation BPB. Some phrases look more locally fluent,
while speaker transitions, grammar and invented words still fail. This contrast
is a reason to assess prediction, repetition, memorization and coherence
separately. The evidence does **not** establish that the approved qualitative
generation objective has been fully achieved.

The predetermined 5,000-step diagnostic also produces more recognizable
ROMEO/JULIET/Nurse exchanges than the early best greedy king loop, but includes
“bauteous”, “comonduit”, “orcess”, “offection” and “sunderstand”. It mixes plays
and relationships and repeats short conversational turns. Its validation BPB
is worse than both the selected F01 state and E02. Thus this intermediate state
does not resolve the prediction-versus-presentation trade-off; its complete
50-slot output grid remains available rather than being treated as a new pilot.

For a bounded demonstration, use the **best checkpoint, temperature 0.7**, with
the existing fixed sampling protocol and explicit limitations. This is a
qualitative presentation recommendation from the predetermined generation grid,
not a new decoding method or a test-based change to checkpoint selection.
No top-k, top-p, repetition penalty or filtering was introduced.

## Literal training-text reproduction

The train-only overlap audit covers a predeclared set of 20 outputs: both
checkpoints, all five prompts, greedy and temperature 0.7/seed 42. It excludes
each supplied prompt and searches only the immutable training partition.
The complete substrings and source offsets are in
`fixed20_training_overlaps.json`.

Best-checkpoint longest exact matches range from 20 to 33 characters, mostly
common speaker labels and short phrases. Such short matches neither prove nor
exclude memorization. At the final checkpoint, the JULIET greedy output contains
a **371-character exact training passage beginning at offset 923,562**, including
Provost and ABHORSON dialogue about an execution. The final “To be, or not to be”
temperature-0.7/seed-42 output contains a **136-character exact LADY ANNE passage
at offset 260,534**.

These are observed verbatim reproductions of training passages. Alongside the
training/validation divergence, they support concern about memorization in the
late state. They are not an estimate of a corpus-wide memorization rate and do
not prove that all generated phrases were memorized. Test text was not searched
to select examples, change the architecture or improve generation.

## Final test and scientific boundaries

Validation BPB selected the update-1,750 checkpoint. Its identity was frozen
before the single reserved full sliding test assessment: **test BPB 2.24333983**,
loss 3.48200190 and token perplexity 32.524768. It scored 55,514 original bytes
after the common 256-byte conditioning prefix. The test partition did not affect
training, checkpoint selection or model changes; it was not rerun for tuning.

Test and validation results concern different contiguous source text. Their
difference is not a significance test or evidence that the implementation is
invalid. The conditional metric, tokenizer boundaries, finite model context and
reset local positions are documented in the evaluator and report.

BPE changes effective byte context and source-byte exposure. Width changes
parameter count, head dimension and MLP dimension. Depth changes capacity and
compute. Matching seed 42 does not make initial tensors equal across those
shapes. F01 jointly changes these factors and uses a 10,000-update authorized
budget; comparisons against 5,000-update pilots must retain those confounders.
The original 14 architectures are different conditions, not independent
replications of one model. No cross-seed standard deviation can be reported.

## Acceptance versus research success

Engineering acceptance requires the independent archive and CPU receipt, the
fresh CUDA full-state receipt, exact fixed-grid reproduction, all 40 checkpoints
and 41 immutable snapshots, plus the report and reviewable PR. The accompanying
report is gated on those genuine receipts; a good unit-test count alone is not
acceptance evidence.

Successful execution and preservation do not imply that every scientific goal
succeeded. Early held-out prediction improved relative to E02, continued
training harmed it, greedy repetition remained severe at the selected state,
and malformed text/invalid bytes limit the demonstration. These negative
findings belong in the final presentation alongside the measured BPB gain.
