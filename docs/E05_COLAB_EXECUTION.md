# E05 seed42 Colab execution and recovery

This execution is scoped to the user's explicit E05 seed42 authorization for
exactly 5000 completed optimizer updates. The canonical E05 registry remains
unchanged. No additional research seed, configuration, factor combination or
continuation is authorized by these files.

## Provenance

Train the installed shared package from clean reviewed main
`31e974b23b7e73b2f48b080b36865608efb879b8`, with scientific source hash
`cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6`.
The separately versioned orchestration repository supplies `relu_audit.py` and
`e05_colab.py`; it does not replace the package model/training/evaluation loop.
Record both commits in the launcher and receipts.

The full numerical audit independently constructs E01 and E05 at seed42 and
compares all initial parameter tensors. All four actual MLPs must be ReLU,
192→768→192, with 1,804,800 total parameters. It distinguishes ReLU/GELU
numerically, checks actual negative-preactivation gradients, causal invariance,
shifted targets and finite CUDA forward/backward. A failure blocks research.

## Environment and durable gate

The thin notebook provisions isolated Python3.11.14, torch2.5.1+cu124/CUDA12.4
and the exact46-package E01 freeze on a Tesla T4. Native Colab packages are not
used for research. Keep `MPLBACKEND=Agg` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Use a fresh exclusive directory under
`MyDrive/Nano-Language-Model/E05_seed42/pilot_<uuid>`. Read actual Drive quota;
FUSE `disk_usage` is not the account quota. Preflight verifies data/tokenizer
hashes, writes using staged rename, executes separate labeled smoke updates,
verifies all exported manifests/file hashes, restores into a fresh directory,
executes actual CUDA inference and smoke state continuation. A mounted path
alone never satisfies this gate.

## Research and measured timing

The research phase invokes `nano_lm.train.train` with the unchanged `TrainSettings`:
seed42, AdamW3e-4, weight decay0.01, constant LR, clipping1.0, float32 CUDA,
microbatch4, accumulation8,4096 target tokens/update; validation250, checkpoint500,
full sliding batch32, conditioning256bytes, no evaluation cap. Stop exactly5000.

Thin timing hooks forward unchanged checkpoint/export/journal calls and record
actual overhead. They do not alter numerical arguments, optimizer behavior,
validation, sampling or RNG. The engine journal/checkpoint histories are the
scientific record. Preserve failures and attempted events; an interrupted
research process may resume only from a compatible hash-verified immutable state.
Never relaunch from initialization as a claimed exact continuation.

## Postprocessing and independent verification

Only run the finish phase after authoritative completion. Verify all snapshots,
restore into a fresh directory and rehydrate all20 historical checkpoint states
solely from their hashes. Reuse the reviewed CUDA model/optimizer/scheduler/scaler/
RNG state roundtrip in a fresh subprocess, with no optimizer update. Generate
both complete grids using `configs/generation_protocol.json` unchanged.

Activation diagnostics use four predetermined128-character training windows
in evaluation mode, after training. Inactivity in this finite512-position sample
is not permanent dying-ReLU evidence. No held-out selection or optimization
occurs during diagnostics.

Package every checkpoint, full history, metadata, tokenizer, generations,
figures and recovery receipts. Verify ZIP paths/CRC/entry hashes locally, publish
with staged rename, read back the archive hash, and split into ordered200MiB
parts for independent connector retrieval. Verify part hashes, exact reconstructed
archive SHA256, all entry hashes, checkpoint inventory/counters and best/final
inference with `verify_e05_archive.py`. Keep weights/data/private file references
outside GitHub.

`compare_relu.py` compares all20 measured E01/E05 validation endpoints and their
matching character loss/BPB units. Runtime observations retain Kaggle/Colab
hardware/driver confounds. `write_e05_report.py` renders the report only after
independent archive verification and requires separate genuine qualitative
observations. The full unedited grids remain available; no sample rewriting or
favorable selection is permitted.

## Collaboration

The dedicated E05 branch, notebook, report and Drive namespace isolate this work
from the primary researcher's E06 branch and E01–E04 artifacts. Draft review PR10
and a coordination comment on E06 PR9 record that scope. Complete the report and
verification receipts before marking the E05 PR ready; do not merge it.
