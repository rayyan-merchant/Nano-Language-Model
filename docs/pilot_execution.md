# Authorized E01 seed42 pilot execution

On 2026-10-12 Rayyan explicitly authorized E01 seed42, exactly 5000 completed
optimizer updates, on legitimate Kaggle rayyannmerchant. This is a scoped
single-account exception; study-wide NO-GO persists. No other seed/configuration
or 10k extension is authorized.

Canonical E01/model/optimizer are unchanged: character vocabulary65, parameters
1804800, width192, layers4, heads6, context128, sinusoidal, GELU, MLP4x,
dropout0.1; AdamW3e-4 constant, weight decay0.01, clip1, float32. Effective4096
predicted tokens/update = microbatch4 * context128 * accumulation8. Full sliding
validation every250, complete55514-byte suffix after256 conditioning bytes,
batch32. Checkpoint every500 plus each validation/best boundary remains intact.

The scripts/pilot builder requires an explicit committed source SHA, generates a
unique private notebook, clones that clean commit and provisions isolated
Python3.11.14/torch2.5.1+cu124. All transitive packages are pinned to the recorded
qualified freeze; only the editable project's Git SHA differs. GPU/disk/session
checks and exact reference-manifest comparisons precede a separate one-step
development smoke. No test-set likelihood is evaluated. No existing smoke run
is relabeled research.

run.py calls the existing nano_lm.train.main CLI parser/engine with explicit
arguments. Timing hooks only forward original checkpoint/export/journal calls;
no training math, sampling, initialization or evaluation semantics change. The
research engine reseeds after preflight. Analysis reads real checkpoint history
and generates the entire fixed protocol grid from best/final weights. Generation
uses a private sampling generator and never touches training checkpoint state.

Each committed checkpoint triggers the existing verified immutable snapshot
mechanism. Retain all snapshots and raw checkpoint history in notebook output.
A compact archive contains the complete research run/all checkpoint files,
plots, machine-readable histories, generations and supplemental evidence once.
An external manifest records every archive file hash and archive SHA256.

Kaggle batch execution automatically ends to publish its saved output version;
publication status and independent retrieval/checksum/inference verification
are mandatory before declaring artifacts durable. Do not manually cancel the
job after training or assume working storage persists. If interrupted, preserve
the saved outputs and restore the latest verified compatible snapshot. Never
restart under the same run ID or automatically fall back to CPU training.

Run completion is asserted only after5000 committed history rows and20 actual
full validation measurements; processed targets/source bytes must both equal
20480000. No early stopping or automatic extension. Single-seed conclusions
remain provisional and cannot establish all14 configurations' convergence.
