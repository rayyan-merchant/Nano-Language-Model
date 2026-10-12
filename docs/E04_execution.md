# E04 seed42 scoped execution — 2026-10-12

User explicitly authorizes E04 alone, seed42, exactly5000 completed updates on
rayyannmerchant's qualified Kaggle T4 GPU0. No other seeds/configurations/10k
continuation. E01/E02/E03 evidence and canonical configurations remain unchanged.
Reviewed main before work:7b4d922c50e9f1663fff8f02829eb95545b9eb7f.
Scientific core matches cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6.

Sequential plan: audit learned embeddings and source/config equivalence; test
thin-wrapper/builder changes and commit a clean source; run independently reseeded
CUDA preflight (forward/backward/reload/export/Agg plots/generation), then canonical
5000-update research CLI; monitor full validation; publish original saved outputs;
independently retrieve/archive/hash/CPU-infer on fresh spacious D storage; fresh
compatible GPU full-state restoration with zero updates; generate three-way
character-position comparisons and a separate contextual E02 BPB table; report
all genuine results/negative observations/failures; PR, no merge.

Locked learned position table128x192 adds24576 parameters (1.3617%) to1804800:
total1829376. The canonical model initializes that table before later layers,
consuming extra RNG draws. Same seed therefore preserves token embedding initial
weights but not identical later-layer weights. This existing behavior is not
modified; it is a reproducibility/attribution limitation alongside extra capacity.
No sinusoidal addition or RoPE active. Audits intercept actual embedding addition,
poison unused sinusoidal buffer, prohibit rotation, check gradients/causality and
cropped generation indices. No core correction was required.

Architecture width192/layers4/heads6/context128/MLP4/GELU/dropout.1; character65.
AdamW3e-4 constant/wd.01/clip1,float32,micro4/acc8,target4096,eval250/checkpoint500
plus existing validation saves. Full sliding256 conditioning bytes/55514 target
bytes. No test likelihood. Isolated Python3.11.14/torch2.5.1+cu124/CUDA12.4/
NumPy1.26.4 and exact qualified transitive freeze; native runtime untouched.

Review means explicit source/diff/CLI checks, independent numerical audits and
passing tests/CI; no human review approval is claimed. All execution changes must
be committed before submission. E03 reporting utilities remain historical; new
E04 reports never overwrite them. All snapshots and original journal stay durable.
Publication success is established by actual terminal saved output and subsequent
independent checksum verification, not mere working-directory export.

Status before execution: preflight/tests pending; no research result claimed.
