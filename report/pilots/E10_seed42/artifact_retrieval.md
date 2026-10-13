# E10 verified artifact retrieval

Research saved version1: https://www.kaggle.com/code/rayyannmerchant/nano-e10-pilot-20261013-403eb78d

Fresh CUDA state-restoration saved version1: https://www.kaggle.com/code/rayyannmerchant/nano-e10-recovery-20261013-e2fb429a

Private account: `rayyannmerchant`. Use private authentication; never share tokens.

Archive: `D:/Nano-Language-Model-artifacts/E10_published_20261013_403eb78d_ranges/E10_seed42_pilot.zip`

Size272,568,662 bytes; SHA256 `f20b2394f1d0a99f132caa90a3148cb08a03fb7682b4507c4dce206015d9bf8b`.

Verified extraction: `D:/Nano-Language-Model-artifacts/E10_published_20261013_403eb78d_ranges/verified`.

Run: `E10_seed42_a15bcd2b54f346fcb25830c70aa10b7b`.
Best and final are both `runs/E10_seed42_a15bcd2b54f346fcb25830c70aa10b7b/checkpoints/step_005000.pt`.
Checkpoint SHA256 `b4f0e17427158fb45920c6f2b34a2c270b84a16ac0f58096af68b9e3294a9819`.

All20 resumable checkpoints, histories, metadata, tokenizer/data manifests,
100 fixed generations and9 original PNG/vector-PDF figure pairs are in the ZIP.
All21 immutable snapshots remain in the saved notebook under `pilot/snapshots`;
the ZIP excludes duplicate snapshot files but retains their complete index.
Fresh CUDA verified snapshots and restored exact state with zero extra updates.
Independent local CRC/file-set/all59 file hashes and CPU inference passed.

For a new independent download, use new empty paths:

```powershell
kaggle kernels output rayyannmerchant/nano-e10-pilot-20261013-403eb78d/1 -p NEW_DOWNLOAD --file-pattern 'E10_seed42_pilot.zip|artifact_manifest.json'
.venv/Scripts/python.exe scripts/pilot/inspect_archive.py NEW_DOWNLOAD/E10_seed42_pilot.zip NEW_DOWNLOAD/artifact_manifest.json NEW_VERIFIED
```

Two actual CLI transfers ended in IncompleteRead and are retained in the initial
and `_retry1` D-drive directories. An official streaming attempt preserved45MiB
under `_streamed`. `download_kaggle_archive.py` then used the assigned account's
official output API and four bounded HTTP ranges to recover remaining bytes;
exact206/Content-Range/size and complete archive SHA256 were required. Signed
URLs and credentials were not logged. Receipt: `evidence/download_receipt.json`.
The utility runs with the separate installed Kaggle-tool Python3.13, not the
locked research environment; it does not install or modify research dependencies.
All range attempts/prefixes remain preserved. No earlier experiment was overwritten.

Optional CPU-audit submission was rejected before execution by title validation;
it used no compute, was unnecessary once local recovery succeeded, and its unused
source was removed. No optional snapshot-duplicate download was performed locally.
See `evidence/commands.json` and `execution_review.json` for actual commands/failures.
Restoring artifacts does not authorize additional optimizer updates.
