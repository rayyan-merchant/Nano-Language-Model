> Historical execution workflow. Commands and notebook paths refer to the
> pinned trained source in the corresponding [research report](../report/README.md),
> not necessarily the current submission tree. See [historical retrieval](maintenance/README.md).

> Historical execution procedure. Current reports record experiment acceptance; individual earlier attempts may have failed. This is not renewed training authorization. Use [current reports and status](README.md) for actual acceptance and artifact retrieval. Pinned commands/source versions remain preserved below.

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


## E04 seed42 complete — 2026-10-12

Individually authorized E04 learned-position run completed exactly5000 updates
and all20 full sliding validations on rayyannmerchant's qualified T4 GPU0.
Clean trained source b260264521ceb280f356b82ecd8cf4b69cb8af21; unchanged audited
scientific core cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6.
Parameters1829376 (+24576). Best/final5000 loss1.55111692,BPB2.23778869;
all matched E01/E03 validation losses are lower than E04. Extra capacity and
canonical initialization RNG consumption limit pure representation attribution.
Training1001.78457s,validation413.79723s,pipeline1766.15717s,peak152.0874MiB.
Original Kaggle5e7b13ef/v1 and fresh recoveryd31e262c/v1 are COMPLETE.
Independently retrieved407744239-byte archive SHA256
60b25eeab43606cedf1ef1e4ab86f9dfbbe10ccc366c1bac290015996e3a86fe:
58 files verified;20 checkpoints and21 snapshots preserved. Independent CPU
inference and exact fresh CUDA full-state roundtrip/generation passed,zero
extra optimizer updates. Read-only monitoring connections dropped/reconnected;
research journal has no failures/resumes. D recovery prevented C-space problems.
Windows38 passed/1 CUDA skip; CPU/CUDA13 learned-position checks passed; training
commit CI green before submission. Actual20 checkpoint counters and character
BPB identity independently audited.100 unedited generation rows/50 duplicate
best-final pairs; strong greedy loops/malformed words retained.14 measured figures,
three-way character comparisons and separate contextual E02 BPB summary:
report/pilots/E04_seed42/E04_seed42_analysis.md plus interpretation.md.
E01/E02/E03 evidence and canonical definitions unchanged. E04 seed42 alone is
complete; broader three-seed study remains separately gated. No other experiment,
seed or10k extension launched. PR8 is reviewable and is not merged by this task.
