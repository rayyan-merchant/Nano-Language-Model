> Historical execution workflow. Commands and notebook paths refer to the
> pinned trained source in the corresponding [research report](../report/README.md),
> not necessarily the current submission tree. See [historical retrieval](maintenance/README.md).

> Historical execution procedure. Current reports record experiment acceptance; individual earlier attempts may have failed. This is not renewed training authorization. Use [current reports and status](README.md) for actual acceptance and artifact retrieval. Pinned commands/source versions remain preserved below.

# E10 seed42 execution record — 2026-10-13

Authorization is one E10 seed42 run, exactly 5,000 completed optimizer updates.
No other configuration, seed, hybrid or extension. Main was synchronized at
a6b28d7d97f9739dddd24ec4c7c3765bc65265dc, after E09 PR13 was merged. The
collaborator's E07 PR11 remains untouched. Scientific source and locked configs
match audited E01; core hash is
cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6.

Only mlp_expansion changes from 4 to 2: each block maps 192→384→192, with
148,032 MLP parameters instead of 295,872. Total parameters are 1,213,440,
591,360 fewer than E01 (32.765957%). Attention, four pre-LayerNorm blocks,
GELU, sinusoidal positions, dropout 0.1 and all training/evaluation controls remain
unchanged. Different tensor sizes consume initialization/dropout RNG differently;
matching seeds do not imply identical whole-model weights. The private data
sampler remains independently seeded. Architecture and capacity change together.

Sequential gates: 14 CPU numerical MLP checks; full applicable tests; committed
thin wrapper and successful CI; clean pinned Kaggle source and exact qualified
Python3.11.14/torch2.5.1+cu124/CUDA12.4/NumPy1.26.4/freeze on Tesla T4 GPU0;
data/tokenizer/partition hashes; actual CUDA audit and full suite; independent
real-data development smoke with save/load/export/restore/ZIP/Agg/generation;
full reseeding and unchanged shared CLI research training; 20 validations scoring
55,514 bytes after 256 conditioning bytes; 20 resumable checkpoints and 21
immutable snapshots; saved publication, independent archive integrity/CPU
inference, fresh zero-update CUDA restoration; 100 fixed-protocol generations;
matched E01 comparison, measured figures, report and reviewable PR.

AdamW LR0.0003 constant, weight decay0.01, clipping1, float32, microbatch4,
accumulation8, 4,096 targets/update. Stop exactly at5,000; no test likelihood.
The legitimate rayyannmerchant OAuth account was verified, with26.01h GPU quota
remaining. Large local artifacts remain on D: through the existing outputs
junction. Output paths refuse reuse. Any scientific or provisioning gate failure
blocks training rather than silently changing methodology. Results are pending
until genuine execution and independent recovery complete.

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
