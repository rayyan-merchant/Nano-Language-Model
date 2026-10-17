> Historical execution workflow. Commands and notebook paths refer to the
> pinned trained source in the corresponding [research report](../report/README.md),
> not necessarily the current submission tree. See [historical retrieval](maintenance/README.md).

> Historical execution procedure. Current reports record experiment acceptance; individual earlier attempts may have failed. This is not renewed training authorization. Use [current reports and status](README.md) for actual acceptance and artifact retrieval. Pinned commands/source versions remain preserved below.

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

## E09 seed42 eight-block research run — completed 2026-10-13

One explicitly authorized E09 seed42 run completed exactly 5,000 optimizer
updates and 20 full sliding validations on rayyannmerchant Kaggle Tesla T4 GPU0.
Fresh initialization; only n_layers4->8 differs from E01. Parameters3,584,256,
+1,779,456 (98.596%) versus E01; depth, capacity and computation change together.
Clean trained source c555269ad6de293c016d88beec2e1437e4289e3f; audited scientific
hash cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6 unchanged.
Python3.11.14/torch2.5.1+cu124/CUDA12.4/NumPy1.26.4 and qualified dependency
freeze matched. Dataset, partitions and character tokenizer matched reference.

CPU and CUDA15 depth checks passed, including independent Q/K/V/sqrt32/causal
attention references and gradients across all eight blocks. Final Windows suite:
46 passed,1 CUDA-only skip122.72s. Actual Kaggle CUDA suite47 passed32.43s.
Real-data CUDA smoke save/load/export/restore/ZIP/Agg/generation passed before
full reseeding. All20 checkpoint optimizer/scheduler counters, journals, BPB
and E01/E08/E09 configuration/environment/sampling/coverage comparability passed.

Best4750 loss1.45571123/BPB2.10014737; final5000 loss1.46725990/BPB2.11680859.
Training1811.126s, full validation739.049s, clone-to-archive pipeline3117.872s;
peak allocated199.370MiB. Best BPB lower5.145% versus E01 and2.113% versus E08,
at97.568%/19.834% extra training time. Single seed; no significance/depth-only
causality. Late improvements diminish, with a final uptick, not sustained
observed deterioration. Greedy loops and malformed language remain.

Private research saved version1: rayyannmerchant/nano-e09-pilot-20261013-b9ea5610.
All20 resumable checkpoints and21 immutable snapshots published. Independently
retrieved794894378-byte archive SHA256
293aaecba3bd794fcd41a58dff08791a63d897bf20a5b913a75b2adbaa3a7b21 passed ZIP CRC,
exact entry set and59 file hashes. Best/final CPU inference passed. Fresh CUDA
saved version1 rayyannmerchant/nano-e09-recovery-20261013-66f79ad5 restored exact
model/optimizer/scheduler/scaler/RNG/sampling/history at5000 with zero extra updates;
its checkpoint hashes and64-token samples match independent CPU recovery.
No research failure/resume event. Local monitoring reconnects/OAuth401 were
resolved through legitimate existing-account refresh, without restarting training.

Complete100-sample fixed grid,18 measured PNG figures,9 vector PDF figures,
all20 matched comparisons, machine-readable results, exact commands and detailed
interpretation are in report/pilots/E09_seed42/E09_seed42_analysis.md.
Large artifacts remain onD and published Kaggle outputs, excluded fromGit.
Historical E08 placeholder E09 values remain untouched; the current three-way
comparison uses the actual recovered E09 measurements.

E08 PR12 was already owner-merged before preparation; this task did not merge it
or claim a formal human review. Main was refetched and remained
 d1ca09df3d20478aabcdd3b4999f8cef6e0d5782. E09 PR13 is for review, no automatic
merge. Earlier reports/configurations and E07's separate work are unchanged.
This scoped run is complete. The broader three-seed study remains separately
gated/NO-GO; no other experiment, seed or10k extension was started. Recommendation:
retain5k screening and consider separately approved selected10k extensions and
additional seeds;5k does not prove convergence.
