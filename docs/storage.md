# Current artifact access and preserved storage procedures

Accepted per-run publication/retrieval/CPU and compatible CUDA recovery receipts are indexed in [the project map](README.md). Earlier Drive/Kaggle-PENDING statements below are original task snapshots; persistence was subsequently tested for the recorded completed runs. This is not a guarantee for a future account/session.

F01 [retrieval](../report/final/artifact_retrieval.md) and B01 [retrieval](../report/balanced/artifact_retrieval.md) identify exact immutable versions/checkpoints/hashes; each pilot has its own record. Cleanup downloads, rewrites and deletes no external model artifact. Large weights remain outside Git.

---

# Persistent storage and interruption recovery

## Actual E02 Colab/Drive evidence — 2026-10-12

The scoped E02 seed42 pilot verified a mounted writable Drive directory, actual
Drive account quota, a smoke snapshot export/checksum/fresh restore/inference and
CUDA continuation before training. The 5k run preserved all twenty validation
states in twenty-one immutable snapshots. The original local runtime was absent
after completion; all artifacts were recovered in a fresh matching runtime with
zero additional updates. Every optimizer/scheduler counter and archive file hash
was independently checked, including downloaded archive reconstruction and CPU
inference. This is evidence for this account/run, not a blanket claim for all cloud
storage accounts. See [the E02 report](../report/pilots/E02_seed42/E02_seed42_analysis.md).

The archive manifest initially used `atomic_json`, whose hard-link publication is
unsupported on Drive FUSE. Postprocessing now uses the storage module's staged
rename publisher. Preserve the failed temporary file/log as evidence. The complete
477,310,968-byte ZIP exceeds this connector's 268,435,456-byte download limit;
ordered parts under the limit preserve the original archive SHA256. Verify part
hashes, concatenate in numeric order, and verify the full archive and entry hashes.

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

Actual Windows connector audit: a nested export failed because its temporary
filename exceeded the supported path length. Packaging the remote evidence into
a ZIP and exporting to a short local filename succeeded; SHA256, ZIP integrity
and three snapshot manifests verified after download. The runtime was released
after verified publication, and a downloaded GPU checkpoint loaded on CPU. Keep
archives and separately verified manifests; see docs/evidence/gpu_export_validation.json.
