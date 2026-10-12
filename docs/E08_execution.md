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
