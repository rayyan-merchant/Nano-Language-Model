# Actual commands, checks and delivery boundaries

All values refer to genuine F01 execution or separately labeled development
evidence. No research update followed the authorized 10,000-step target.

## Source and preflight checks actually executed

- `git fetch origin`, clean-source checks and reviewed-main/PR17 verification.
- `scripts.final.protocol.provenance` Git-blob comparison: all original scientific
  files and locked historical configs unchanged; core fingerprint recorded.
- `python -m scripts.final.audit --root . --dataset ... --device cpu --output ...`:
  actual combined CPU math/gradient/causal/BPB audit receipt.
- Initial full CPU suite: 76 passed, one failed, one skipped. The failed
  exactly-14-config assertion was updated for the explicitly approved F01 file;
  every original override remains checked. Corrected suite: 77 passed, one
  CUDA-only skip. Independent float32 reference tolerance was corrected without
  changing model mathematics; deliberately wrong references still fail.
- Qualified Kaggle development CUDA suite: **78 passed in 39.52 s**, from its
  actual bootstrap log. Real dataset smoke, 10-update benchmark/full validation,
  immutable export, independent download and CPU/fresh CUDA continuity completed.
- Later CPU helper suite: **79 passed, one CUDA-only skip, 943.04 s** measured
  elapsed duration. This slow local run is not GPU or research runtime.
- Final formatting/byte/generation regression checks: **3 passed in 2.61 s**;
  final link/hash/fence protection regression: **1 passed in 4.23 s** before
  fresh report rendering. Local Matplotlib/pyparsing deprecation warnings were
  recorded; headless plots rendered successfully.
- CI succeeded for execution source06291e1 (run38489000521), launcher9a93b7a
  (run38489784207), and helperd4a9329 (run38497274185). Final delivery CI must be
  checked on its actual pushed commit; prior green runs are not substituted.

## Actual research execution

The exact canonical CLI argument list is preserved byte-for-byte in
`evidence/research/execution_command.json`. It invokes `python -m nano_lm.train`
with F01/seed42/10,000/research/CUDA/float32/AdamW3e-4/wd.01/clip1/micro4/
4096targets/eval250/checkpoint500/full sliding/batch32/prefix256/no evaluation cap.
Its pre-run test-reservation flag describes the training phase, not the later
single frozen reserved assessment in `reserved_test_evaluation.json`.

Research kernel `rayyannmerchant/nano-f01-research-20261014-05b5949a/1` is COMPLETE.
Fresh recovery `rayyannmerchant/nano-f01-research-recovery-20261014-d22a4fd8/1`
is COMPLETE. Current-source core identity was rechecked after all report changes.
There are 10,000 sequential journal updates, no failure/resume records, 40
checkpoint counters and 40 complete sliding validations. No replay was found.

## Actual independent retrieval and postprocessing commands

```text
kaggle kernels status rayyannmerchant/nano-f01-research-recovery-20261014-d22a4fd8
kaggle kernels output rayyannmerchant/nano-f01-research-recovery-20261014-d22a4fd8 -p D:/Nano-Language-Model-artifacts/F01_research_CUDA_recovery_20261014_d22a4fd8 --file-pattern (recovery.json|generation_reproduction.json|environment.json|hardware.json|failure.json)$
python scripts/pilot/download_kaggle_archive.py --kernel rayyannmerchant/nano-f01-research-20261014-05b5949a/1 --manifest D:/Nano-Language-Model-artifacts/F01_research_small_20261014_05b5949a/artifact_manifest.json --output D:/Nano-Language-Model-artifacts/F01_research_published_20261014_05b5949a --ranges
python -m scripts.final.inspect --archive D:/Nano-Language-Model-artifacts/F01_research_published_20261014_05b5949a/F01_seed42_final.zip --manifest D:/Nano-Language-Model-artifacts/F01_research_published_20261014_05b5949a/artifact_manifest.json --output D:/Nano-Language-Model-artifacts/F01_research_published_20261014_05b5949a/verified --dataset data/processed/tiny-shakespeare-v3
python -m scripts.final.report --root . --verified D:/Nano-Language-Model-artifacts/F01_research_published_20261014_05b5949a/verified --manifest D:/Nano-Language-Model-artifacts/F01_research_published_20261014_05b5949a/artifact_manifest.json --cuda-receipt D:/Nano-Language-Model-artifacts/F01_research_CUDA_recovery_20261014_d22a4fd8/final/recovery.json --generation-receipt D:/Nano-Language-Model-artifacts/F01_research_CUDA_recovery_20261014_d22a4fd8/generation_reproduction/generation_reproduction.json --output D:/Nano-Language-Model-artifacts/F01_preparation_20261014/final_delivery_build_v2 --kernel rayyannmerchant/nano-f01-research-20261014-05b5949a
python -m scripts.final.delivery --root . --output D:/Nano-Language-Model-artifacts/F01_preparation_20261014/final_delivery_build_v2
```

Quoted paths were used in PowerShell; `OMP_NUM_THREADS=1` was set for local
inspection/testing/reporting. Official Kaggle SDK download used its Python3.13
tool environment; project inspection used local `.venv` torch2.5.1 CPU. Recovery
and exact100-slot reproduction used the isolated qualified Python3.11.14 CUDA
package against execution source06291e1. No new test likelihood was computed.

Local archive inspector passed CRC, exact82-file inventory, safe paths and every
SHA256; all40 optimizer/scheduler/history counters, sampler RNG/byte exposure/
coverage and genuine CPU inference passed. Separate CPU/CUDA greedy probe text
and checkpoint checksum equality checks passed for best/final. Fresh CUDA
verified all41 native snapshots and all six full-state fields with zero extra
updates. All100 fixed strings reproduced; strict byte errors remained recorded.

The first report render succeeded. Regeneration into that existing directory
then failed with FileExistsError from the deliberately no-replace artifact
writer. This was a local reporting failure, not a scientific checkpoint failure.
The final corrected renderer requires a fresh output directory. It and the
supplement ran successfully in `final_delivery_build_v2`; reviewed files were
promoted to report/final only after checking that existing raw receipt, summary,
validation and generation hashes remained identical. Historical E01–E14 files
were not edited. Markdown links, architecture arrows and actual comparison plots
were inspected; `git diff --check` passed after whitespace correction.
Scoped Git attributes preserve raw F01 receipts, generation strings and CSV
line endings across Windows/Linux checkouts. Staged generation.json/md,
summary.json and validation.csv hashes were explicitly compared with the
published manifest and all match; raw CSV was restaged under the exact-byte rule.

## Measured scopes

Training4,269.206 s; full validation945.441 s; checkpointIO15.380 s;
immutable export80.292 s; journal/fsync36.637 s. Measured wrapper/archive
components sum6,317.442 s; inner timings must not be added again to this sum.
Independent download11,213.970 s; complete local archive/CPU inspection205.529 s.
Provisioning, queue/publication latency and complete task wall time were not
measured as a single duration. No invented aggregate replaces them.

The notebook's immutable archive preserves native measured analysis; later
research interpretation, comparative charts and documentation are lightweight
Git deliverables. Large checkpoint files, complete journals, transfer pieces and
raw datasets remain outside Git on D: and the legitimate owner's saved Kaggle
outputs. No secret, signed download URL or credential is committed.
