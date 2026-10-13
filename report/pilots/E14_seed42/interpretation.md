# E14: observations, interpretation and recommendation

## Hypothesis before execution

Increasing width192→240 could improve next-character modeling through additional
capacity, or increase cost without a useful predictive gain. Width also changes
head dimensions32→40, MLP hidden dimensions768→960, tensor shapes, initialization
RNG consumption/fan-in and sinusoidal frequency spacing. No direction was assumed.

## Observations

E14 has lower full-validation NLL than E01 at all 20 matched scheduled points.
Best BPB2.14189208 at4750 is0.07216585 lower than E01-best2.21405793 (−3.259438%).
Final BPB2.14505608 is0.07094096 lower (−3.201311%). E13/E01/E14 best BPB is
2.32495848/2.21405793/2.14189208 for widths144/192/240. These are measured
single-seed results, not means across training seeds. The twenty points share one
trajectory and held-out corpus; they are not twenty independent replicates.

E14 adds1,004,160parameters (+55.638298%). Training978.158s versus916.708s
(+6.703340%); validation490.769s versus420.908s (+16.597806%). Recorded allocator
peak177.283MiB versus151.712MiB (+16.854464%). Throughput5.111646updates/sec,
20,937.302targets/sec. E13's smaller model trained941.393s, slightly slower than
E01 despite fewer parameters. Parameter count therefore did not scale measured
runtime proportionally on these separate T4 sessions. No FLOP/kernel/energy
profiling or controlled speed repetitions were performed.

The shared telemetry samples cumulative CUDA allocator peaks after updates;
earlier validations contribute, while a distinct final-validation/postprocessing
peak is not measured. The10-step benchmark samples before its sole final
validation. Its155.351MiB reading is not substituted for the actual research
177.283MiB. This scope is common to the recorded full research comparisons.

E14 first reaches or beats E01's final loss at the recorded3750 checkpoint
(loss1.52954764; training736.750s). This is a retrospective observed-point
threshold, not an interpolated crossing, early stopping rule or proof of a
universal sample-efficiency advantage.

Validation decreases1.58088056→1.48683957 across3k→5k (−0.09404099nats).
Successive1k improvements are0.20682089(1k→2k),0.10655250(2k→3k),
0.05768975(3k→4k),0.03635124(4k→5k). Improvement diminishes but remains positive.
Best4750→final5000 loss increases0.00219312nats. That single small deterioration
does not establish sustained overfitting or an asymptotic plateau. Full
dropout-free validation is fixed; variation here belongs to optimization/model
changes, not capped stochastic validation sampling.

Final training loss1.42570889, trailing100 mean1.41597478. Global pre-clipping
gradient median0.49777606,p950.53976347,maximum2.05958056;4/5,000 updates exceeded
clipping threshold1(0.08%), versus7/5000forE01. Norms remain finite; these global
statistics do not diagnose per-layer vanishing/exploding gradients. No research
failure/resume/replay is recorded; physical and authoritative research updates
are5000. Separate development/test work is excluded.

## Genuine generation: strengths and failures

All 100 unedited best/final slots are preserved. Best4750 and final5000 are
different weights; both complete grids are real outputs from our own model.
Five best greedy examples in the main report are the predetermined prompts,
not a favorable subset. `generation_quality.json` audits every setting.

Observed best greedy repeated8-character-window fraction0.387149, versus
E01's0.522892 and E13's0.763052. E14-final is0.446586. Repetition decreases in
this descriptive comparison but remains severe: all five best greedy prompts
converge toward phrases about “the prince”, “the courts”, “the world” and a
SICINIUS turn. More varied word formation does not yield coherent dialogue.
“I will not the grace of the courts” illustrates a malformed clause; repeated
“the prince of the prince” has no useful semantic progression.

Sampling reduces local repetition: best mean fractions0.002410/0.001339/0.0
at temperatures0.7/1.0/1.2, respectively. Low repetition is not a coherence
score. Predetermined seed 42 samples contain internal malformed words such as
“distressived”, “appaint”, “profide”, “promismal”, “brunishments”, “medditation”
and altered speaker labels. Higher-temperature samples are more varied but
often less grammatical. Valid-looking line breaks and labels do not establish
correct speaker identity or multi-turn consistency.

The literal prompt prefix is retained, but responsiveness is shallow. “To be,
or not to be” often continues with “d” to form “bed”; different speaker prompts
fall into shared tails. Terminal incomplete words are expected under the fixed
256-character stopping limit and are not counted as internal spelling failures.
No generated text was corrected by an external model.

Ten predetermined best outputs (five greedy plus temperature0.7/seed 42 for all
five prompts) have longest exact training-text overlaps14–24characters, mostly
ordinary phrases or speaker-label formatting. This does not prove memorization
or rule it out. No test text or sample cherry-picking was used.

## Interpretation and conclusion

For this seed, corpus and budget, wider capacity improves conditional held-out
likelihood at moderate measured training-time cost, with more parameters and
memory. Lower teacher-forced loss does not establish fluent autoregressive text.
The result is consistent with a width/capacity benefit but cannot isolate width,
head dimension, MLP capacity or initialization independently. Same sampling
windows/RNG/byte union were independently verified; global initialization and
dropout RNG consumption vary mechanically with tensor shapes.

One seed provides no variance, standard deviation, equivalence or significance
estimate. Separate sessions, execution conditions and the telemetry scope limit
resource interpretations. No test-set selection or other unapproved run occurred.

## Recommended next stage

Keep5,000 updates as the initial screening budget. E14's late gains diminish, and
one small final fluctuation is insufficient evidence to change the common budget
or call the model conclusively overfit. Prioritize matched approved training
seeds to test whether the width trend persists. Then consider approved10k
continuations of selected configurations, with E01 retained as a control.
E14 is a reasonable candidate for an extension because late validation still
improves; this is a proposal, not evidence of its10k outcome. Use the verified
final5000 checkpoint's complete state and constant LR. No extension, hybrid,
additional configuration or training seed has been launched.
