# B01 published artifact retrieval and zero-update recovery

Saved research notebook version1:
https://www.kaggle.com/code/rayyannmerchant/nano-b01-research-20261014-33317a00

Independent fresh CUDA recovery version1:
https://www.kaggle.com/code/rayyannmerchant/nano-b01-research-recovery-20261014-d2c9b763

Use legitimate authorized account access. Do not rerun the research notebook,
initialize another trajectory or resume past the frozen4,500 stopping decision.
The source is c2cf1c165d09e0fc8f64a2df714755a4d910d469. Later report/transport
helper commits are separately identified and do not alter the trained source.

## Actual independently recovered locations

```text
D:\Nano-Language-Model-artifacts\B01_research_small_20261014_33317a00\artifact_manifest.json
D:\Nano-Language-Model-artifacts\B01_research_published_20261014_33317a00\B01_seed42_final.zip
D:\Nano-Language-Model-artifacts\B01_research_published_20261014_33317a00\download_receipt.json
D:\Nano-Language-Model-artifacts\B01_research_published_20261014_33317a00\verified\independent_recovery.json
D:\Nano-Language-Model-artifacts\B01_research_published_20261014_33317a00\verified\runs\B01_seed42_c1b6b391cd9b476994325a85c264493a\checkpoints\step_003000.pt
D:\Nano-Language-Model-artifacts\B01_research_published_20261014_33317a00\verified\runs\B01_seed42_c1b6b391cd9b476994325a85c264493a\checkpoints\step_004500.pt
D:\Nano-Language-Model-artifacts\B01_research_CUDA_recovery_20261014_d2c9b763\balanced\recovery.json
D:\Nano-Language-Model-artifacts\B01_research_CUDA_recovery_20261014_d2c9b763\balanced\generation_reproduction\generation_reproduction.json
```

Full archive604,825,667bytes, SHA256
`41742da357777a6aee40bd7bc4a4063ee1e3d07f94b94e2344f082252a467edf`.
It contains all18 actual research checkpoints, metadata/journals/histories,
stopping receipt, complete100-slot generations, raw token diagnostics, figures
and native snapshot index:96 exact files. The37 immutable native snapshot
directories are separately preserved in Kaggle saved outputs, not duplicated
inside the ZIP. Each snapshot retains canonical latest/best states plus complete
metadata/history at that boundary; earlier checkpoint inventory is preserved in
the run and archive. Fresh GPU recovery actually verified all37 native manifests.

Reports and independent recovery receipts were created after original training
publication. They are delivered through PR20; the original immutable training
archive is not rewritten to insert later interpretation or recovery evidence.

## Reproducible independent retrieval

Use fresh uniquely named destination folders. On this Windows host, current
KaggleCLI/SDK runs under the previously installed uv Python3.13 tool environment.
The model inspection runs under the project's separately installed CPU venv.
Never print private authentication tokens or signed output URLs.

```powershell
kaggle kernels output rayyannmerchant/nano-b01-research-20261014-33317a00/1 -p NEW_MANIFEST_FOLDER --file-pattern '^artifact_manifest.json$' --page-size 200 -q
SDK_PYTHON scripts/pilot/download_kaggle_archive.py --kernel rayyannmerchant/nano-b01-research-20261014-33317a00/1 --manifest NEW_MANIFEST_FOLDER/artifact_manifest.json --output NEW_DOWNLOAD_FOLDER --ranges
PROJECT_PYTHON -m scripts.balanced.inspect --archive NEW_DOWNLOAD_FOLDER/B01_seed42_final.zip --manifest NEW_MANIFEST_FOLDER/artifact_manifest.json --output NEW_VERIFIED_FOLDER --dataset data/processed/tiny-shakespeare-v3
```

The downloader requires exact206/Content-Range/part sizes, retains incomplete
attempt evidence and verifies whole object SHA256 before final rename. Native
inspection verifies CRC, safe paths, exact inventory/all hashes/all optimizer and
scheduler counters, full journal/stopping replay, independent sampler RNG and
original-byte union, then genuinely generates from recovered best/final states.
The actual transfer took396.989s and independent CPU audit60.398s; these are
transfer/recovery timings, not research training time or portability guarantees.

## Actual fresh CUDA recovery scope

The submitted `scripts.balanced.workflow --mode research-recovery` imported the
clean pinned package, matched the approved isolated dependency freeze, mounted
the published research version read-only, checked archive/snapshot integrity,
and restored into a fresh writable directory. It reproduced every model,
optimizer, constant scheduler, disabled AMP scaler, Python/NumPy/torch CPU/CUDA
RNG, private sampling generator and complete history field exactly at4500→4500.
The actual receipt reports zero optimizer updates,37 snapshots,96 archive files,
and100 reproduced best/final fixed strings. Independently compared256 generated
token IDs per slot also match, including invalid byte outputs. This is more than
model-weight loading alone.

Do not submit another recovery notebook simply to read these receipts. Any later
training or additional model/seed requires new explicit authorization.
