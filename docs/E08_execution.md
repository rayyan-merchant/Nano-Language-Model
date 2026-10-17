> Historical execution workflow. Commands and notebook paths refer to the
> pinned trained source in the corresponding [research report](../report/README.md),
> not necessarily the current submission tree. See [historical retrieval](maintenance/README.md).

> Historical execution procedure. Current reports record experiment acceptance; individual earlier attempts may have failed. This is not renewed training authorization. Use [current reports and status](README.md) for actual acceptance and artifact retrieval. Pinned commands/source versions remain preserved below.

# E08 seed42 execution plan -- 2026-10-13

Scope: one explicitly authorized six-block run, seed42, exactly5000 completed
optimizer updates. No other configuration/seed or10k extension. E07 remains
the second researcher's Colab work. E06 PR9 was already merged by the owner;
main synchronized at fad0afe2fc6259bd5fd2ddb5f004ce57e87bd34c. No formal GitHub
review submission was returned; human review is not invented. Scientific source
and canonical configurations are unchanged from the audited E01 commit.

Sequential gates: numerical six-layer audit and full applicable CPU suite;
commit/review thin wrapper and verify CI; qualified isolated Kaggle profile and
strict canonical source/data/tokenizer guards; full applicable Kaggle pytest,
actual CUDA depth audit; independent one-update development training/validation,
checkpoint reload/export/fresh restore/ZIP and Agg/generation; larger-checkpoint
storage budget; reseeded canonical researchCLI5000updates/20 validations;
saved output publication and independent download/hash/CPU inference; fresh
compatible CUDA full-state roundtrip with zero extra updates; genuine analysis,
E01 comparison, fixed100-sample grid, figures and research report; reviewable PR.

Only n_layers4->6 changes. Parameters2694528 (+889728), character65, width192,
heads6/dimension32/context128/sinusoidal/GELU/MLP4/dropout.1 unchanged.
AdamW3e-4 constant/wd.01/clip1,float32,micro4/acc8,4096 targets/update.
Validation every250, preserving all20 validation states and21 snapshots.
Conditioning256 original validation bytes, scored55514, full sliding. No test
likelihood. Same scientific hash cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6.

Development audit records seed42 repeated-construction equality and observed
shared E01/E08 parameter equality rather than assuming identity. Every block's
preLN/residual computation is compared with a manual module-level reference;
independent storage, per-block finite/nonzero gradients, causality, dropout and
cropped inference are verified. Numerical smoke gradients do not establish
research vanishing/exploding behavior. Extra capacity and computation confound
depth; one seed cannot establish significance.

Existing verified E06 saved-output recovery is the durable workflow reference.
Preflight requires free space exceeding100 actual smoke-checkpoint equivalents
plus2GiB margin (20 states,21 best/latest snapshots, compact archive, overhead).
Local large outputs remain onD via the existing outputs junction; fresh artifact
directories cannot overwrite earlier evidence. Research metrics remain pending
until actually executed and independently verified.

## E08 seed42 completed and independently recovered -- 2026-10-13

The explicitly authorized six-block E08 run completed exactly5000 optimizer
updates and20 full sliding validations on TeslaT4 GPU0,Python3.11.14,
torch2.5.1+cu124/CUDA12.4/NumPy1.26.4,float32. Clean trained source
72fdd4b64610e8e6df178e7fd452ec9429faab8f; scientific hash
cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6.
Only n_layers4->6 differs;2694528 parameters (+889728 capacity confound).
CPU/CUDA13 depth checks passed. Before launch Windows44 passed/1 CUDA skip;
actual full Kaggle suite45 passed33.84s, including CUDA AMP-resume tests.
Best4750 loss1.48713993/BPB2.14548940; final5000 loss1.48858993/BPB2.14758132.
Training1511.357s, validation593.279s, clone-to-archive pipeline2602.147s;
allocated peak166.104MiB. Lower best BPB than E01 by3.097% costs64.868% more
training time. One seed does not establish significance or depth-only causality.

Research private saved notebook version1:
rayyannmerchant/nano-e08-pilot-20261013-79fc614d. All20 resumable states and21
immutable snapshots published. Independently downloaded598768138-byte archive
SHA256 f1071463a41c18954cb8eb177b9e8a24ba9a692077ddb97e438c8af49d881fb4
passed59 file hashes/ZIP checks and best/final CPU inference. Fresh saved CUDA
recovery rayyannmerchant/nano-e08-recovery-20261013-75b12285 verified exact model,
optimizer/scheduler/scaler/RNG/sampling state at5000, zero additional updates.
All20 checkpoint counters and character BPB independently audited. No research
failure/resume event.100 unedited generations and15 measured PNG figures retained;
greedy loops/malformed words remain despite improved validation likelihood.
Full report and interpretation: report/pilots/E08_seed42/E08_seed42_analysis.md.
Large artifacts remain onD and saved Kaggle outputs, excluded fromGit.

E06 PR9 was already owner-merged before source synchronization; no formal review
submission is claimed. E08 PR12 is for review and not merged by this task.
Earlier reports/configurations and E07 Colab work are unchanged. No other
configuration, seed or10k extension was launched. E09 depth-framework metrics
remain null/not-run. This scoped E08 run is complete; the broader three-seed
study remains separately gated/NO-GO. Final test evidence is in the E08 report.
