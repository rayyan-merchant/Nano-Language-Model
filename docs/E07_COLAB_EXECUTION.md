> Historical execution procedure. Current reports record experiment acceptance; individual earlier attempts may have failed. This is not renewed training authorization. Use [current reports and status](README.md) for actual acceptance and artifact retrieval. Pinned commands/source versions remain preserved below.

# E07 seed42 Colab execution

Authorized scope: exactly 5000 completed optimizer updates, seed42, E07 only.
Fresh initialization; no extensions or other configurations. Primary researcher
owns E06. E07 uses a dedicated branch, notebook, report and Drive directory.

Trained source is clean reviewed main `5bb03b18943e7c45a305ce1d78b3446c31829ae7`.
Scientific source hash remains
`cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6`.
Only the locked E07 head override is used: eight heads, dimension24, width192,
four blocks, GELU, sinusoidal positions, character65, context128, dropout0.1,
MLP192→768→192 and 1804800 parameters. Shared scientific modules are unchanged.

`heads_audit.py` independently checks projection shapes, sqrt24 against a manual
reference and incorrect sqrt32/sqrt192 alternatives, per-head softmax/masking,
dropout train/eval, head merge, finite/nonzero projection gradients, future-token
invariance, shifted targets, generation cropping, and E01 initialization equality.
CPU checks are correctness evidence; research measurements require qualified CUDA.

Provision isolated Python3.11.14, PyTorch2.5.1+cu124, CUDA12.4 and exact46-package
freeze on Tesla T4. Leave Colab native packages untouched. Set MPLBACKEND=Agg,
OMP_NUM_THREADS=1 and CUBLAS_WORKSPACE_CONFIG=:4096:8. Pin the committed E07
orchestration independently of the clean installed main package.

Read actual Drive account quota, mount the approved Drive and create a fresh
`Nano-Language-Model/E07_seed42/pilot_<uuid>` destination. Run the `preflight`
phase of `scripts/pilot/e07_colab.py`: qualified environment, CUDA audit, existing
suite plus focused test, exact data manifest, staged-rename write probe, labeled
smoke checkpoints/export/manifest verification/fresh restore/inference and full
state smoke continuation. Check conservative artifact budget before research.

`research` calls unchanged `nano_lm.train.train`, float32 AdamW3e-4, constant LR,
wd.01, clip1, micro4/accum8, 4096 targets/update, full sliding validation every250,
checkpoint cadence500 retaining all validation states. It stops at5000. Timing
hooks forward unchanged calls without RNG/numerical changes. No test likelihood.

`finish` verifies all immutable snapshots, recovers all20 checkpoint states,
audits optimizer/scheduler/history and full evaluation byte coverage, executes a
fresh compatible CUDA state roundtrip with zero updates, generates the complete
canonical best/final grid and genuine figures, and publishes verified archives
and ordered parts using the tested Drive publisher. Independently download parts,
verify with `verify_e07_archive.py`, then build matched comparison and report.

Preserve both the clean reproducible launcher and exact executed Colab notebook
with outputs. Never represent the launcher as the executed evidence. Keep weights,
dataset and archives outside GitHub. Open review PR; no merge.

## Authorized Kaggle fallback

After Colab refused GPU allocation, the user requested Kaggle and signed into
verified account `syedarijaali`. Native CUDA probe attached two Tesla T4 GPUs
(driver580.178.04); native Python3.13/PyTorch2.11 remains unqualified. User then
instructed completion of the same E07 task on this available platform. The frozen
scientific configuration and exactly5000-update scope remain unchanged.

`scripts/pilot/e07_kaggle.py` uses the committed E07 audit and same training/finish
functions. Its platform preflight reproduces the exact approved environment,
data, tests and complete-state smoke workflow. Publish a saved Kaggle output
version, independently retrieve/hash the smoke archive, then mount that saved
output in a new notebook and execute compatible fresh CUDA state recovery before
research. Saved output versions provide platform persistence; working-directory
snapshots alone are explicitly ephemeral. Final verified archives/parts also go
to the authorized Google Drive destination through the connected file uploader.
No Drive mount or continuous Drive export is falsely claimed for Kaggle.
