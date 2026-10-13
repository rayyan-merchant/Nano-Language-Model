# E11 interpretation, separated from measurements

## Observations

E11 has lower validation loss/BPB than E01 at all twenty matched scheduled
updates. Its best and final checkpoint is update5000: loss1.49301304,
BPB2.15396251, perplexity4.45048484. E01's best BPB2.21405793 occurred at4750.
The best-score difference is−0.06009542BPB/−2.7143%; the final-score difference
is−0.06203453BPB/−2.7994%. Parameters are unchanged at1,804,800.
These are observed single-seed differences, not significance estimates.

E11 reaches an observed validation loss below E01's final loss at3750, but has
already spent1398.53 training seconds there, versus E01's916.71 total training
seconds. Earlier crossing in optimizer updates does not imply earlier crossing
in time. The comparison file records retrospective scheduled-point thresholds;
no interpolated crossing is presented as an observation.

Training took1868.69s,103.85% more than E01. Full validation took267.25s,
36.51% less. Peak allocated memory was116.146MiB,23.44% less. Training
throughput was2.67567updates/s and10959.52targets/s. Separate T4 sessions
and no repeated timing prevent precise hardware-efficiency generalizations.

Validation loss continued decreasing at every scheduled point. Improvements
per1000 updates decreased from0.20047 in1k–2k to0.10825 in2k–3k,
0.05699 in3k–4k and0.03162 in4k–5k. The3k–5k decrease was0.08861.
This shows diminishing but continuing improvement, not complete convergence.
There was no observed sustained validation deterioration. The same-step final
training loss1.41863 is noisy; the final100-step mean is1.46373. A widening
dropout-on sampled-training versus dropout-off sliding-validation gap alone
does not establish overfitting or memorization.

Aggregate pre-clipping gradient median/p95 were0.46728/0.50535; maximum1.94567.
Six of5000 updates exceeded clipping norm1, versus seven inE01. All recorded
losses/gradient norms were finite. No research failure or resume was recorded.
There is no longitudinal per-layer gradient evidence for vanishing/exploding
gradients; the development checks establish connectivity only.

All five fixed greedy prompts collapse into variations of “world of the world”,
sometimes repeating LADY GREY dialogue blocks. Mean repeated8-gram-position
fraction is0.81526 versus E01-best0.52289. The complete100-row generation grid
is preserved. Best and final are the same weights, so the two50-output labels
duplicate deterministic settings; they are not100 independent samples.

Temperature sampling greatly reduces this repetition metric: E11 means0.00348,
0.000535 and0 at temperatures0.7,1.0,1.2 respectively. This does not prove
coherence. Outputs include malformed words such as “devery”, “accontent”,
“contruction”, “forwul” and “Farewelch”, grammar such as “I cannot not”, and
weak continuity despite recognizable character labels and line breaks.
ROMEO/JULIET at temperature1/seed42 have identical continuation text; shared
sampling seeds themselves are a confound when interpreting similar trajectories.
Poor prompt discrimination is an observation, not proof that context64 is its
sole cause. No long-range-dependency benchmark was executed.

For the predetermined five greedy and five temperature0.7/seed42 best samples,
longest exact training substrings range15–20 characters. These short overlaps
include ordinary phrases and speaker labels. They do not establish extensive
verbatim memorization; nor does this limited descriptive check exclude it.

## Interpretation

Under this fixed-token protocol, the shorter-context run achieved better
held-out conditional likelihood but worse greedy repetition and approximately
double training time. It offers lower activation memory and validation cost,
not an observed training-speed improvement. Predictive likelihood and free
generation quality should be reported separately.

The intended change alters conditioning information. Every sliding-validation
prediction sees at most64 previous characters rather than128, while the common
256-byte exclusion only fixes scoring boundaries. Training also averages targets
at within-window positions1–64 rather than1–128. The corresponding mean available
within-window conditioning lengths32.5 and64.5 are derived protocol quantities,
not measured source exposure or FLOPs. Overlapping windows stay inside training;
they do not leak held-out text.

E11 samples64 windows/update rather than32, requiring16 microbatches rather than8.
Both expose20,480,000 targets, but E11 draws320,000 windows versus160,000 and
has a different private RNG trajectory and target union. Exact replay verified
each separately. Initial trainable weights are identical under seed42; sampling,
dropout and gradient-grouping paths thereafter differ. Equal token budgets are
therefore not equivalent optimizations or computations.

## Hypotheses for later verification

More numerous shorter windows may alter gradient variability and useful
training-context distributions. Additional small-microbatch overhead may help
explain slower training despite shorter attention tensors. These are plausible
mechanisms, not isolated causal findings: no kernel-level profiler, alternative
grouping experiment or additional seed was executed. Better likelihood cannot
be attributed exclusively to the reduced conditioning limit.

## Conclusion and recommended next stage

Keep5k as the initial screening budget, but prioritize separately approved10k
extensions of selected configurations after multi-seed evidence. E11 is still
improving at5k;5k is a common screening endpoint, not proof of convergence.
Repeat authorized seeds before selecting context64 as a replacement for128,
and retain both likelihood and unedited generation diagnostics. E12/context256
remains unrun and unauthorized; the framework contains no invented E12 metrics.

The experiment's measured training, publication, archive recovery, CPU inference,
twenty checkpoint counters and sampling audits are verified. Fresh compatible
CUDA full-state restoration must be judged separately from those successes.
Its latest status is recorded in evidence/cuda_recovery.json. Do not call the
whole acceptance checklist complete while that gate is pending. No further
research updates or extension have been executed.
