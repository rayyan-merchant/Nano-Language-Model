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
