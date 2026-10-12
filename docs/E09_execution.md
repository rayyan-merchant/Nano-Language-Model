# E09 seed42 execution plan — 2026-10-13

User authorization is one E09 seed42 run, exactly5000 completed optimizer updates.
No other configuration, seed, combined architecture or extension. E07 is the
second researcher's Colab work and remains untouched. E08 PR12 was already merged
by the owner at d1ca09df3d20478aabcdd3b4999f8cef6e0d5782; no formal GitHub review
submission was returned, so a human review is not claimed. This merged main is
the preparation base. Scientific source/configurations match audited E01 exactly.

The only override is n_layers4->8:3584256 parameters,1779456 extra versus E01
(98.59574468%), and889728 extra versus E08. Width192/heads6/head32/context128,
character65/sinusoidal/GELU/MLP4/dropout.1 stay unchanged. AdamW3e-4 constant,
weight decay.01,clip1,float32 CUDA,micro4/acc8,4096 targets/update. Full sliding
validation every250 scores55514 original bytes after256 conditioning bytes;
test likelihood remains reserved. Fresh initialization, never prior weights.

Sequential gates:15 CPU numerical depth checks and full applicable tests;
commit/inspect thin wrapper and verify CI; pinned clean Kaggle source with exact
qualified Python3.11.14/torch2.5.1+cu124/CUDA12.4/NumPy1.26.4/freeze,TeslaT4 GPU0;
canonical data/tokenizer/partition hashes; actual CUDA depth audit and full pytest;
one-update development real-data training/save/reload/export/fresh full-state
restore/ZIP CRC/Agg plotting/generation; measured larger-checkpoint free-space
budget; reseeded shared trainingCLI5000 updates/20 validations; complete journals,
20 resumable states and21 snapshots; actual saved output publication, independent
download/CRC/all-file hashes/CPU inference; fresh CUDA exact-state restoration
with zero additional updates; genuine100-sample grid; three-way E01/E08/E09
comparisons, measured figures, research report, reviewable PR. No automatic merge.

The numerical audit includes independent Q/K/V reshape, sqrt32 scaling, causal
softmax/value aggregation/merge references; all eight independent preLN residual
blocks and finite/nonzero gradients, dropout and cropped generation. Smoke gradients
establish connectivity, not longitudinal vanishing/exploding-gradient conclusions.
Seed42 shared initialization equality is recorded rather than assumed. Depth,
capacity, initialization RNG consumption and compute change together.

Qualified quota was checked:3.09h used/26.91h remaining. Earlier short E09 runtime
projection is only an estimate; final training and validation times will be
measured separately. Storage gate requires100 actual smoke-checkpoint equivalents
plus2GiB margin. Large local artifacts remain onD via the existing outputs junction.
Unique paths refuse reuse. Existing E08 saved-output recovery is the durable
reference; execution is blocked by any source/environment/data/numerical/storage
gate failure. Results are pending until genuine execution and independent recovery.
