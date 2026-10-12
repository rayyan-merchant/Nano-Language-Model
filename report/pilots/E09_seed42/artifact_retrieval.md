# Verified E09 artifact retrieval

Research saved version 1: https://www.kaggle.com/code/rayyannmerchant/nano-e09-pilot-20261013-b9ea5610

Fresh CUDA zero-update recovery version 1: https://www.kaggle.com/code/rayyannmerchant/nano-e09-recovery-20261013-66f79ad5

Both are private to the legitimate `rayyannmerchant` account. Use that account's
private authentication; never share tokens. All large evidence is excluded from Git.

Archive: `D:/Nano-Language-Model-artifacts/E09_published_20261013_b9ea5610/E09_seed42_pilot.zip`

SHA256: `293aaecba3bd794fcd41a58dff08791a63d897bf20a5b913a75b2adbaa3a7b21`

Size: 794,894,378 bytes. Exact entry set, ZIP CRC and 59 file hashes passed.

Verified extraction: `D:/Nano-Language-Model-artifacts/E09_published_20261013_b9ea5610/verified`

Run: `E09_seed42_380197e606df4082b22b5ddff146946d`.
Best: `runs/E09_seed42_380197e606df4082b22b5ddff146946d/checkpoints/step_004750.pt`.
Final: `runs/E09_seed42_380197e606df4082b22b5ddff146946d/checkpoints/step_005000.pt`.

The ZIP contains all 20 resumable checkpoints, complete journals/metadata,
100 generations, nine PNG/vector-PDF figure pairs and the snapshot index.
All 21 immutable snapshots remain in the saved output under `pilot/snapshots`;
the ZIP excludes their duplicate files. Fresh CUDA recovery verified saved
snapshots and restored full state with zero additional updates. Restoring does
not authorize an extension or another experiment.

For independent retrieval, choose new empty directories:

```powershell
kaggle kernels output rayyannmerchant/nano-e09-pilot-20261013-b9ea5610/1 -p NEW_DOWNLOAD_DIRECTORY --file-pattern 'E09_seed42_pilot.zip|artifact_manifest.json'
.venv/Scripts/python.exe scripts/pilot/inspect_archive.py NEW_DOWNLOAD_DIRECTORY/E09_seed42_pilot.zip NEW_DOWNLOAD_DIRECTORY/artifact_manifest.json NEW_VERIFIED_DIRECTORY
```

The verified machine-readable manifest is `evidence/artifact_manifest.json`.
Use `--file-pattern '^pilot/snapshots/'` in a new directory if all duplicate
snapshot files are needed. That optional duplicate-file retrieval command was
not executed locally; the actual saved-input CUDA recovery verified snapshots.
See `evidence/commands.json` for commands actually executed in this task.
