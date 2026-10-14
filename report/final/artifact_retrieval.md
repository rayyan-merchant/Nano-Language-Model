# F01: published artifacts and independent recovery

Research output: [rayyannmerchant/nano-f01-research-20261014-05b5949a](https://www.kaggle.com/code/rayyannmerchant/nano-f01-research-20261014-05b5949a), **saved version 1**.
The notebook and artifacts may require the legitimate owner's Kaggle sign-in.
Private access is not equivalent to a publicly downloadable model release.

The saved output includes `F01_seed42_final.zip` (2,648,712,125 bytes), its
`artifact_manifest.json`, and native `final/` checkpoint/snapshot/analysis files.
Archive SHA256:

```text
96de06da78ecb1416c8029eaf792003991a45a3dfa160eade43157bbe6ed78ab
```

The ZIP contains all 40 research checkpoint states once, append-only journals,
full measured history, tokenizer/data/configuration/environment metadata,
generation grids, plots and snapshot index. The 41 immutable snapshots retain
their own manifests and duplicate resumable state in the notebook's **native
saved outputs**. They are intentionally not duplicated inside the ZIP. Retrieve
native snapshots as well when the immutable snapshot inventory itself is needed.
The fresh CUDA recovery verifies every native snapshot file; the independent
ZIP receipt verifies all archive files and all 40 checkpoint counters.

Best checkpoint inside the ZIP:

```text
runs/F01_seed42_3ab6cfae8a944d0e988aead87fc4a2fd/checkpoints/step_001750.pt
ff30ec7b18237ff3e0578155f4c5c3f9958dd32b2e458d1a43d2fa61ecc3266f
```

Final checkpoint:

```text
runs/F01_seed42_3ab6cfae8a944d0e988aead87fc4a2fd/checkpoints/step_010000.pt
bea7125eb36f536aaa02df5f46912faa3855d783a2819ba36c8586e540c972e2
```

Large local evidence is on D:, outside Git:

```text
D:/Nano-Language-Model-artifacts/F01_research_published_20261014_05b5949a/
D:/Nano-Language-Model-artifacts/F01_research_CUDA_recovery_20261014_d22a4fd8/
```

The first folder retains the archive, immutable downloaded range pieces,
`download_receipt.json` and `verified/`. The second holds independently
retrieved fresh CUDA state and 100-slot exact generation-reproduction receipts.
Do not delete earlier experiments or partial transfer evidence while relocating
storage. The workspace `outputs` junction already directs large output to D:.

## Reproduce retrieval and verification

Authenticate privately as the legitimate owner; never paste tokens into chat or
commit credentials. Use unique, empty destination folders. The normal CLI can
retrieve saved outputs; for the large archive the committed bounded-range helper
provides retries and a mandatory whole-object SHA256 check:

```powershell
python -m nano_lm.prepare --output data/processed/tiny-shakespeare-v3
python -m nano_lm.prepare --output data/processed/tiny-shakespeare-v3 --validate
kaggle kernels output rayyannmerchant/nano-f01-research-20261014-05b5949a -p FRESH_SMALL --file-pattern 'artifact_manifest.json$'
python scripts/pilot/download_kaggle_archive.py --kernel rayyannmerchant/nano-f01-research-20261014-05b5949a/1 --manifest FRESH_SMALL/artifact_manifest.json --output FRESH_ARCHIVE --ranges
python -m scripts.final.inspect --archive FRESH_ARCHIVE/F01_seed42_final.zip --manifest FRESH_ARCHIVE/artifact_manifest.json --output FRESH_VERIFIED --dataset data/processed/tiny-shakespeare-v3
```

The downloader uses the official authenticated output URL, four bounded 8 MiB
ranges and exact HTTP 206/Content-Range/size checks; the signed URL is never
recorded. A failed transfer can reuse completed pieces in a **new** destination
with `--resume-pieces PREVIOUS/pieces`; metadata and final whole-object SHA256
must still match. A successful transfer alone does not establish checkpoint
acceptance; the independent inspector verifies safe paths, exact inventory,
ZIP CRC, each SHA256, counters/history, RNG sampler replay and CPU inference.

Fresh qualified CUDA recovery:
[nano-f01-research-recovery-20261014-d22a4fd8](https://www.kaggle.com/code/rayyannmerchant/nano-f01-research-recovery-20261014-d22a4fd8), version 1 COMPLETE.
It restores the final checkpoint's model, optimizer, scheduler, disabled AMP
scaler, Python/NumPy/torch CPU and CUDA RNG, sampling generator, byte coverage
and full training history. All six state fields match after save/reload;
10,000 → 10,000, **zero additional optimizer updates**. Best/final actual CUDA
inference and all 100 fixed generation strings reproduce after recovery.

Frozen training checkout: `06291e11630eaaa316873740e8223b1fc00ba3cb`.
Pure inference diagnostic helper: `d4a9329459f23c1521148c25076045b4b0467ed0`,
blob SHA256 `cd0bc4c30cbabf0f7e899fc3597c2077bf3ea0d512835882fffe5b2a8d7f01ab`.
The helper runs against the frozen training package, not a silently upgraded
scientific implementation. Reserved test likelihood was assessed once in the
research notebook after selection freeze; retrieval does not rerun it.

Small raw receipts are in `evidence/research/`; complete metrics/configuration
and plots are linked from [the report](F01_10000_analysis.md). The final narrative
and comparative figures are committed Git evidence created after publication;
the published archive preserves the original measured research outputs and
native analysis. No checkpoints, raw datasets or full archive are committed.
