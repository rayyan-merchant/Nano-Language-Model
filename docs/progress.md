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
   report including at least two failure modes and 5â€“10 minute recorded demo.

Other collaborator: fetch canonical main, create a feature branch, install pinned
requirements, run pytest and the documented smoke/CLI; report hardware and errors.
Do not start GPU runs or change locked configs. Coordinate ownership through PRs
and committed registry after runtime profiling. No automatic cross-chat sync.
