# E14 durable artifacts and exact recovery

## Immutable saved versions

- Research version 1: https://www.kaggle.com/code/rayyannmerchant/nano-e14-pilot-20261013-82e48728
- Fresh final-state CUDA recovery version 1: https://www.kaggle.com/code/rayyannmerchant/nano-e14-recovery-20261013-1bd9f987
- Development publication version 1: https://www.kaggle.com/code/rayyannmerchant/nano-e14-development-20261013-fa3959be
- Development fresh CUDA recovery version 1: https://www.kaggle.com/code/rayyannmerchant/nano-e14-development-recovery-20261013-c25df910

These private saved outputs require legitimate `rayyannmerchant` access. Do not
use another account's credentials. No signed download URLs or credentials are
stored in Git. Use the pinned version and verified hashes, not an unversioned
working-directory assumption.

## Research archive and states

Archive `E14_seed42_pilot.zip`: **623,713,697 bytes**.
SHA256: `e41393c4a97c2ec565558df4ca0997d74275cf62e69c57b466c4794855b26fd7`.
Run ID: `E14_seed42_91b80cdf4163420f8767b58010eb4e9a`.
Configuration hash: `09eebce689f9a3fb71e59ee6402a2dd46ec0a42b29fbe2e83c6fb2e3cb19f5ac`.
Frozen trained source: `ab1da4e05005413807d20802fff4153d97ebfa10`.

Best checkpoint `step_004750.pt`, SHA256
`61d53c46b5b8f30bf58fc0ade1625460b03eada4c488dc27b6e02b71796ab300`.
Final checkpoint `step_005000.pt`, SHA256
`f16a09408c934b2e87690675765562238339900bb76cd074cbd74cbec812b339`.
All twenty validation states at 250-step intervals are preserved with complete
model/optimizer/scheduler/scaler/RNG/sampling/history state. The native saved
output additionally preserves all twenty-one immutable snapshots. The compact
archive stores each distinct checkpoint once plus complete journals, metadata,
generation, analysis figures and the snapshot index. Full inventory and every
file SHA256 are in `evidence/artifact_manifest.json`.

Verified Windows root:
`D:/Nano-Language-Model-artifacts/E14_published_20261013_82e48728/verified`.
Checkpoints are under
`runs/E14_seed42_91b80cdf4163420f8767b58010eb4e9a/checkpoints/`.
The archive, bounded-transfer parts and download receipt remain alongside that
root. Native snapshot manifests are separately retained under
`D:/Nano-Language-Model-artifacts/E14_snapshot_manifests_20261013_82e48728`.
Fresh CUDA receipt:
`D:/Nano-Language-Model-artifacts/E14_cuda_recovery_20261013_1bd9f987/recovery_audit/recovery.json`.

## Verified recovery procedure

Use `scripts/pilot/download_kaggle_archive.py --ranges` with the official Kaggle
API, saved version 1, and this genuine manifest. It uses bounded 8 MiB chunks,
four streams, exact HTTP range/size checks and complete SHA256 verification.
`--resume-pieces` can reuse complete prior ranges in a new unique output folder;
never overwrite failed-attempt evidence. Then run `inspect_archive.py` against
the complete ZIP/manifest in a fresh extraction directory. It checks CRC, safe
paths, exact inventory, all hashes, counters and actual best/final CPU inference.

The already completed fresh CUDA notebook mounted the original saved research
output, verified actual snapshot files, restored the final checkpoint's complete
training state and saved an exact roundtrip. All six state groups matched at
5,000→5,000, with no additional optimizer updates. Best/final CUDA inference
matched recovered CPU inference. All twenty optimizer/scheduler counters were
also independently checked locally. This does not claim that an unapproved
5k→10k training continuation was executed.

## Lightweight report and figures

The full analysis, interpretation, operations, machine-readable comparisons,
100-slot genuine generation grid, 20 PNG/20 PDF figures and checksummed receipts
are published through the dedicated E14 GitHub PR. These final written reporting
files are separate lightweight artifacts; they were not retroactively inserted
into the already immutable research ZIP. Raw datasets, model binaries and full
training journals remain outside Git. Prior experiment archives are untouched.

Development archive SHA256 is
`981ddb740545097312b82660e9b3d521a3e03ae3352983aaa2710e2b61239679`
(282,780,282 bytes). Its independent 81-file/snapshot/CPU/fresh-CUDA receipts
are in `docs/qualifications/evidence/e14_development/`; it is explicitly
development evidence, not another research run.
