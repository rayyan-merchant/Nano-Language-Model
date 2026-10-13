# E13 observations, interpretation and limitations

## Hypothesis

Reducing width192→144 might retain much of the baseline's quality with less
memory/compute, or remove useful capacity. No direction was assumed. The sole
configuration override derives head dimension32→24 and MLP768→576, while also
changing embedding/output dimensions, parameter count, initialization fan-in
scales/RNG consumption and sinusoidal frequency spacing. The algorithms remain
unchanged. This is a width/capacity ablation, not an independent intervention on
parameter count, head size or MLP size.

## Predictive observations

Best=final5000:loss1.61153841,BPB2.32495848,perplexity5.01051355. E13's loss is
higher than E01 at all20matching validation boundaries. Its best BPB exceeds
E01-best by0.11090055/5.0089%; final BPB exceeds E01-final by0.10896144/4.9170%.
There is no preregistered tolerance that establishes predictive equivalence.
Across-seed variance/SD is undefined, not zero; these differences do not establish
statistical significance or universal behavior at other widths/datasets.

Validation decreased at every scheduled point. Loss improvements were0.20766
in1k–2k,0.11981 in2k–3k,0.07998 in3k–4k and0.05407 in4k–5k. The3k–5k decrease
was0.13405; late recorded-point fitted slope−0.06558 per1000updates. These are
observations, not interpolated samples or a forecast for10k. Improvements are
diminishing but continuing; the endpoint does not demonstrate complete convergence.
No sustained validation deterioration or double descent was observed.

Final training loss1.59743 and its final100-update mean1.58467 remain below
validation1.61154. Sampled/dropout-on training and full sliding/dropout-off
validation differ, so this gap is not a direct memorization or overfitting rate.
A smaller model could be capacity-limited or require further optimization;
the5k single-seed trajectory cannot distinguish those hypotheses.

## Resource observations

Parameters fall1,804,800→1,021,824:782,976 removed/43.3830%. Training took941.39s
versus E01's916.71s (+2.6928%), so this run provides no training-speed advantage.
Validation took310.33s versus420.91s (−26.2722%). Peak allocated memory132.291MiB
versus151.712MiB (−12.8018%). Throughput5.31128updates/s and21754.99targets/s.
Measured clone-to-archive duration1625.39s excludes queue,publication,network
retrieval and fresh recovery. Peak allocation excludes reserved/driver memory.

Equal4096targets/update does not mean equal FLOPs. Attention score matrices
retain the same six heads/context128, while projection/MLP dimensions shrink.
More fixed overhead or different CUDA kernel alignment could explain the lack
of proportional speedup, but no kernel profiler or repeated timing was executed.
Those mechanisms are hypotheses. Separate T4 sessions and CPU/platform noise
limit precise attribution of the small training-time difference.

## Sampling and gradient observations

Both runs use32windows/update,8microbatches,160000sampled windows and20480000
predicted target positions/source bytes. Independent private-sampler replay
regenerated each saved RNG and exact target-byte union; they match between runs.
Unique target-byte coverage is1003848 of1003854. Overlapping sampled windows
remain inside training; this is not held-out leakage or uniform epoch traversal.
Global model initialization/dropout RNG trajectories are not assumed equal.

E13's aggregate pre-clipping gradient median/p95/max are0.50862/0.57321/1.56789.
Seven updates exceeded clipping norm1, the same count as E01. All recorded losses
and norms were finite. No longitudinal per-layer gradient measurements establish
vanishing/exploding gradients; numerical CPU/CUDA checks establish connectivity.
The authoritative research history has5000updates,zero failures andzero resumes.
No superseded/replayed research execution occurred. Separate development/tests
are excluded from the research count and training-only time.

## Genuine generation observations

All100unedited fixed outputs are retained in generation.md/json. Best and final
are identical weights, so their50settings duplicate deterministic outputs rather
than100independent draws. Five predetermined greedy examples appear in the main
report, including failures. All five converge to variants of “see the see” and
“That the shall be the see”, with ungrammatical repeated lines.

Mean repeated8-gram-position fraction is0.76305 for E13-best versus0.52289 for
E01-best. Temperature sampling greatly reduces that descriptor:0.000535,
0.003748 and0.001071 at0.7,1.0,1.2. Lower repetition does not establish coherent
generation. Predetermined seed42 outputs include “kingdrace”, “prection”,
“firdsh”, “cunnsafess”, “brungshice”, “To-morrucious” and malformed speaker labels
such as “DUKE VINCA” and “CRIOLANUS”. Common words, line breaks and recognizable
speaker formatting coexist with weak grammar, semantic continuity and prompt
responsiveness. Higher temperature increases variety without fixing these errors.

The fixed five greedy and five temperature0.7/seed42 overlap diagnostics find
longest exact training substrings13–27characters. Ordinary phrases and speaker
labels recur in Shakespeare; this limited indicator neither proves extensive
verbatim memorization nor excludes it. No formal grammar/long-range benchmark
or human preference study was executed. Poor generations alone cannot isolate
width as the cause; capacity, derived dimensions and initialization change together.

## Interpretation and conclusion

Under this seed/data/5k protocol, width144 has a predictive penalty and worse
greedy repetition, with lower allocated memory and validation cost. Fewer
parameters alone do not establish better quality or faster training. The model
may still be useful under tighter memory constraints, but no application-specific
quality tolerance was established and no predictive equivalence is claimed.

Keep5k as an initial screening endpoint rather than a convergence certificate.
Additional separately approved seeds should precede architecture selection;
selected10k continuations could help distinguish optimization limits from
capacity limits. None was launched. The width comparison framework retains
genuineE13/E01 values and E14=null/NOT_AUTHORIZED_NOT_RUN, with intended widths
144/192/240. No E14 results, combined factors or statistical claims are invented.

## Acceptance and operational evidence

PhaseA:E11's original fresh CUDA restoration passed and PR15 was merged before
E13. PhaseB:E13 passed width audits,actual CUDA preflight,all20validations,
independent archive CRC/inventory/all61filehashes,all20checkpoint counters,
21snapshotmanifest checks,CPUinference and fresh full-state CUDA restoration.
Recovered CPU/CUDA best/final generation matches; restoration addedzero updates.
Transport/SSE disconnects were handled separately and did not restart training.
See operations.md and checksummed receipts for exact failure/retrieval details.
