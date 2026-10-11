# Progress

## Foundation implemented

Strict model configuration and 14 canonical experiment definitions; deterministic
character tokenizer/hash; contiguous split and shifted batching; explicit causal
multi-head attention; sinusoidal, learned and RoPE methods; decoder blocks and
GPT loss; greedy/temperature inference; versioned own checkpoints; unique synthetic
CPU smoke runner with JSON logs and optimizer/scheduler/RNG resume; meaningful
CPU tests; thin notebook structure and CPU CI.

## Verification

2026-10-11 Windows Python 3.10.11 / torch 2.5.1+cpu: `python -m pytest -q`
executed 14 tests: 14 passed in 72.88 seconds. After changing checkpoint
publication to atomic create-if-absent, `python -m pytest tests/test_checkpoint.py
-q` executed 2 tests: 2 passed in 49.56 seconds. `python -m pip check` reported
no broken requirements. `git diff --check` passed. E01 architecture check using
an assumed vocabulary of 65 (not a fitted dataset result) produced finite logits
[1,128,65], attention [1,6,128,128], and 1,804,800 parameters.

Installation: isolated CPU torch wheel, pinned numpy/pytest/setuptools and editable
package installed successfully. Exact installed dependency snapshot is recorded.
GitHub CLI is absent; authenticated GitHub connector verified rayyan-merchant
with admin/push permissions. No local Git credential is configured, so publishing
uses the authenticated connector and preserves existing history.

There
are no Tiny Shakespeare experiments, trained research checkpoints, experimental
samples, measured GPU throughput or deployment results.

## Required next gates

1. Dataset download with pinned revision/raw hash, manifest and boundary audit.
2. Train-only BPE with byte-offset mapping, tokenizer/checkpoint compatibility
   and tested same-text BPB with documented conditioning/context limitations.
3. Research train/validation CLI: configurable batch accumulation at ~4096
   tokens/update, metadata schema, validation-selected best/final checkpoints,
   durable export, interruption reconciliation, GPU/AMP/scaler exact-resume gates.
4. Evidence utilities: manual numeric example, mask heatmap, head shape trace,
   position visualization, model counts/diagram, MLM/CLM comparison.
5. Approved baseline pilot, then approved main 42 runs, actual analysis/plots,
   validation-based extensions and five fixed-prompt trained samples.
6. Own-checkpoint FastAPI inference, hosting, running Colab demo, final Medium
   report including at least two failure modes and 5ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“10 minute recorded demo.

Other collaborator: fetch canonical main, create a feature branch, install pinned
requirements, run pytest and the documented smoke/CLI; report hardware and errors.
Do not start GPU runs or change locked configs. Coordinate ownership through PRs
and committed registry after runtime profiling. No automatic cross-chat sync.

## Unified pipeline M1

Single-instance implementation, second member independently verifies/runs the same code.
Pinned original Karpathy Tiny Shakespeare revision and expected raw SHA256; immutable
UTF-8 partitions and manifest; training-only character and byte BPE artifacts; lossless
byte coverage and explicit conditional BPB. Eight preprocessing/BPB tests passed.
Hugging Face TLS failed locally; no certificate verification was disabled.

## Unified pipeline M2

Training/validation engine, exact token-budget accumulation, deterministic RNG,
conditional metrics, source-byte exposure and unique byte coverage, checkpoint
history and append-only attempted-event journal, best/final artifacts and resume
implemented. Tests: 12 passed, 1 CUDA-only test skipped on local CPU. Initial
failures: Windows default encoding in a test (fixed explicit UTF-8), invalid
Adam parameter-equality criterion for near-zero gradient reduction noise
(replaced with direct first-moment gradient and output equivalence assertions).
Real corpus revealed repeated Rust vocabulary-size queries; cached vocabulary
and byte lengths fix the O(tokens*vocab) path. Full acquisition now validates:
1,115,394 bytes; 1,003,854/55,770/55,770 partition bytes; vocabularies 65/1024.

## Unified pipeline M3

Verified immutable storage export/restore, schema/config/lineage/completeness
checking, duplicate-copy handling, cross-member collection, mean/sample-SD
tables and BPB figures, all-config instantiation audit and short benchmark
entry point implemented. Actual Colab T4 CUDA float16 resume test passed:
1 test in 11.93 seconds (Python 3.13.15, torch 2.11.0+cu130). This is a
version-drift smoke verification, not qualification of the pinned research
environment or either independent Kaggle account.

M3 storage/results tests: 3 passed in 11.39 seconds. A Windows long-path
staging-name failure was fixed using short unique temporary names. All 14
models instantiate in the automated audit. Persistent snapshots retain latest
and best checkpoints plus journals, metadata and summaries; corruption fails
verification. Restored snapshots resume without resetting state.

## Unified pipeline M4 and readiness audit

Final stable Windows suite: 31 passed, 1 CUDA skip in 184.73 seconds. Real
Tiny Shakespeare CPU integration passed: original baseline E01 four updates,
E02 two updates, conditional validation, save/reload and own-model generation.
Exported/restored E01 continuation matches uninterrupted model, optimizer,
scheduler, scaler, RNG and loss/validation histories exactly.

Final actual Colab Tesla T4 suite: 32 passed in 50.16 seconds; Python 3.13.15,
torch 2.11.0+cu130. Real-data float16 integration and exact resume passed.
Independent cloud acquisition reproduces the local dataset/tokenizer hashes.
This proves development execution on this GPU, not qualification of the pinned
research profile or two independent Kaggle environments. No long run launched.

EXPERIMENT_READINESS.md records acceptance criteria, actual model counts/shapes,
scientific conditioning/source-exposure limitations and resolved failures. Fixed
generation settings are committed. Results/figures consume genuine metrics,
reject smoke outputs as study evidence and label incomplete seed grids.

**NO-GO** remains until independent Kaggle/environment/storage verification and
all-config GPU/full-validation overhead measurement pass. Review first, qualify
both accounts, then request separate authorization for the 5,000-step pilot.

Post-audit preservation: Colab outputs exported as a 276,461,015-byte archive
after a connector Windows long-path failure; SHA256 and three snapshot manifests
verified on this PC. GPU checkpoint loaded for CPU inference and smoke figures
regenerated. Runtime released after verified publication. Clean Ubuntu GitHub
CI for implementation ffccc1a passed: 31 passed, 1 CUDA skip in 20.24 seconds,
including clean dependency/package installation (run 38158274754). Draft PR #1
is open and unmerged. These additional checks do not remove the Kaggle gates.

Final Windows output correction: real BPE temperature CLI generation exposed
CP1252 printing failure. Explicit UTF-8 stdout and a forced-CP1252 Unicode
subprocess regression resolve it. Stable full rerun: 32 passed, 1 CUDA skip in
227.06 seconds. Real CPU integration rerun at outputs/logs/m4-windows-utf8-cpu
passed exact model/optimizer/scheduler/scaler/RNG and history comparisons again.
Latest evidence is committed separately; T4 results refer to the earlier source
and do not claim execution of this extra output-only CLI regression on a GPU.
