# Persistent storage and interruption recovery

Each run has a UUID identity, immutable metadata, tokenizer/dataset manifests,
versioned step checkpoints, segment summaries and append-only attempted events.
Checkpoint history is authoritative. Events after the latest checkpoint remain
preserved attempts; resume replays from committed optimizer/RNG/sampling state.
Never truncate a journal to make an interrupted run appear complete.

An exclusive .active.lock prevents concurrent trainers. Graceful exceptions log
failure and release the lock. For an expired or killed process, restore the last
verified durable snapshot to a fresh directory instead of deleting evidence or
reusing a possibly active source directory. Exact resume rejects changed data,
tokenizer, model, seed, precision, optimizer, code hash, framework or accelerator.
Local migration for inference is allowed; exact training continuation on changed
hardware/framework is not silently claimed.

```text
python -m nano_lm.storage export RUN DURABLE_ROOT
python -m nano_lm.storage verify SNAPSHOT
python -m nano_lm.storage restore SNAPSHOT RESTORE_ROOT
```

The trainer's --export-root invokes export at committed checkpoint boundaries
and after the completed segment summary. Snapshot directory creation is exclusive;
files are staged and checked, then export_manifest.json is published last as the
completion marker. Export destinations use staged renames in their uniquely owned
directory, so hard-link support is not required on a mounted cloud drive. Incomplete
snapshots have no valid completion manifest and cannot be restored or counted.
Corruption, unsafe paths and existing restore destinations fail explicitly.

Snapshots preserve the latest and validation-best checkpoints plus complete
metadata, journals and available summaries. Each snapshot is immutable; subsequent
snapshots have a different content identity. A newer snapshot may omit historical
weight files that remain in older snapshots; metric history is preserved in the
latest checkpoint. Keep the snapshot for each reporting endpoint (e.g. 5k and 10k).

On Colab mount Google Drive and use a verified MyDrive destination, or export to
the PC through the connector. On Kaggle, keep outputs in /kaggle/working and save
an output version/download verified snapshots before releasing the session.
/kaggle/working itself is ephemeral. Passing a directory path does not prove its
persistence. Test an actual session-end/restore workflow on each account before
research GO. If the destination is inaccessible the trainer fails and retains
local completed checkpoints; do not continue pretending evidence was exported.

Local export/restore/corruption and resumed-history tests are automated. A real
Colab-to-PC export is included in the readiness audit. Mounted Drive and both
Kaggle output-version/expiry workflows remain explicitly pending until executed.
