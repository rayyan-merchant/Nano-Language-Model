> Historical execution workflow. Commands and notebook paths refer to the
> pinned trained source in the corresponding [research report](../report/README.md),
> not necessarily the current submission tree. See [historical retrieval](maintenance/README.md).

> Historical execution procedure. Current reports record experiment acceptance; individual earlier attempts may have failed. This is not renewed training authorization. Use [current reports and status](README.md) for actual acceptance and artifact retrieval. Pinned commands/source versions remain preserved below.

# F01 separately approved execution — 2026-10-14

The user discontinued the three-seed grid and approved one fresh F01 seed42
trajectory, exactly10000 updates. Historical E01–E14 definitions and evidence
remain unchanged. F01 combines canonical BPE1024, RoPE, eight blocks and width240;
heads6/head40, context128, GELU/MLP4x/dropout0.1 and untied LM projection. Actual
parameter count6046560. Training is AdamW3e-4constant, wd.01, clipping1.0, FP32,
micro4/accumulation8,4096predicted BPE targets/update,40full validations and41
expected immutable snapshots. No candidate search, other seed or extension.

## Frozen provenance

- Reviewed pre-F01 main: e4f768d53038662f0f24bcb9cb84a5210ad38a85; E14 PR17 merged.
- Frozen scientific execution: 06291e11630eaaa316873740e8223b1fc00ba3cb.
- Original aggregate: cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6.
- Scientific-only fingerprint: d5b205aecec5913cd11d921c0a84899badd95a014a490a7beaab3d2598c940f8.
- F01 aggregate: 0c3765c7e1d6a9c8698e12e0525496ecd59c6713998ece0d5b3218054c2ab8ff.

The fingerprint maps relative scientific source paths to UTF8 text with universal
newlines; `scripts/final/protocol.py` independently compares every original source
and locked config against reviewed Git blobs. F01.json legitimately changes the
aggregate; no integrity check is disabled. All wrappers invoke `nano_lm.train`.
The notebook separately records its committed launcher and optional inference
diagnostic helper SHA. Those reporting/visibility changes do not alter the frozen
training checkout. Do not merge PR19 automatically.

## Actual preflight evidence

The exact combined CPU numerical audit passed. Corrected source CPU suite77passed,
one CUDA-onlyskip; actual qualified CUDA suite78passed in39.52s. New reporting/
byte-diagnostic delivery suite79passed, one CUDA-onlyskip in943.04s. The initial
exactly14-config-files assertion failed after approved F01 addition and was
updated while retaining every original E01–E14 override assertion. Independent
RoPE division/multiplication references required documented float32 tolerance;
wrong rotation/scaling references still fail. No scientific model fix was needed.

Qualified TeslaT4 GPU0, Python3.11.14, torch2.5.1+cu124/runtimeCUDA12.4,
NumPy1.26.4/tokenizers.21.0 and exact qualified freeze; native environment retained.
Canonical source/partitions/tokenizers/manifest matched. RawSHA86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed;
BPEchecksum2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55.

Development notebook: https://www.kaggle.com/code/rayyannmerchant/nano-f01-development-20261014-f36bc7a7
Fresh development recovery: https://www.kaggle.com/code/rayyannmerchant/nano-f01-development-recovery-20261014-0230bf95
Both version1 COMPLETE. Published134170219byte ZIP independently downloaded,
CRC/path safety/all44files/SHA256verified; archive SHA256
fed6afe393ed333e9549d5160e3b0d528ec294aedfc298bad5885794e9c79479.
CPU inference passed. Fresh CUDA exact model/optimizer/scheduler/disabled scaler/
Python/NumPy/torchCPU/CUDARNG/private sampler/history state passed1→1,0updates.
Smoke-only continuity added1resumed update and2fresh reference updates; equal
tensors/RNG/loss/exposure verified, with0research updates. Native snapshots were
verified after persistent publication; working-directory copies alone were not
accepted. Receipts: report/final/evidence/development.

10-update development benchmark, first2excluded:2.187108updates/s;
21.494138s full validation;73782474byte checkpoint;73851676byte snapshot;
1.346620s export/verify;8.259500s archive/CRC. Training projection4572.248s,
40validations859.766s, pipeline estimate7232.013s with1800s margin. These are
estimates, not research measurements. Conservative storage14690504228bytes versus
20349571072available. Kaggle documents12h CPU/GPU sessions
(https://www.kaggle.com/docs/notebooks); wrapper conservatively budgets10h with
2h provisioning/publication reserve. Quota before research:23.58GPU hours remained.

## Research launch gate — historical pre-completion record

https://www.kaggle.com/code/rayyannmerchant/nano-f01-research-20261014-05b5949a
was submitted once, after all gates passed. No research rerun occurred. Live log
ChunkedEncodingError reconnects, duplicate server log messages and a local tool
connection stall are monitoring events, not evidence of extra optimizer updates.
Checkpoint journals determine authoritative and physical execution accounting.
Research source/optimizer/precision/token budget are frozen. Retain the early best
validation state even when later validation deteriorates; no early stopping or
learning-rate change is permitted. Full acceptance remains pending until10000/
40validations, durable publication, independent all-checkpoint verification,
CPU/freshCUDA restoration, complete fixed generation/reproduction and the final
report pass. Reserved test is evaluated once, after immutable validation selection.
The pending status in this launch record is superseded by the verified closure below.

## Reproducible interfaces

From the repository root, installed project environment:

```text
python -m scripts.final.audit --root . --dataset DATA --device cpu --output AUDIT.json
python -m scripts.final.build_notebook --commit EXACT_SHA --mode development --output FRESH_SUBMIT
python -m scripts.final.build_notebook --commit EXACT_SHA --mode development-recovery --source rayyannmerchant/PUBLISHED_DEVELOPMENT --output FRESH_RECOVERY
python -m scripts.final.build_notebook --commit EXACT_SHA --mode research --source rayyannmerchant/ACCEPTED_DEVELOPMENT_RECOVERY --output FRESH_RESEARCH
python -m scripts.final.build_notebook --commit EXACT_SHA --mode research-recovery --source rayyannmerchant/PUBLISHED_RESEARCH --diagnostic-commit COMMITTED_HELPER_SHA --output FRESH_FINAL_RECOVERY
python -m scripts.final.inspect --archive ZIP --manifest MANIFEST --output FRESH_VERIFIED --dataset DATA
python -m scripts.final.report --root . --verified FRESH_VERIFIED --manifest MANIFEST --cuda-receipt CUDA_RECEIPT --generation-receipt GEN_RECEIPT --output FRESH_REPORT --kernel rayyannmerchant/RESEARCH_KERNEL
python -m scripts.final.delivery --root . --output FRESH_REPORT
```

Building a notebook does not establish publication or successful execution.
Research independently checks the mounted accepted development receipt. Original
source remains frozen even when later report helpers are committed; the optional
helper is fetched from its exact committed blob and SHA256checked, then performs
pure inference using the frozen installed package. All100raw fixed outputs are
reproduced and ByteLevel token bytes strictly diagnosed without repaired text.
Large archives/weights/full journals remain on D: and private Kaggle saved outputs,
outside Git. Save receipt hashes and retrieval/version identities before acceptance.

The report renderer requires a fresh destination and the independently verified
archive download receipt beside FRESH_VERIFIED. Copy the reviewed interpretation
document into FRESH_REPORT before the delivery supplement. Promote reviewed
documents into report/final only after checking immutable receipt and raw sample
hashes. Existing destinations fail rather than silently overwriting evidence.

## Verified research closure — 2026-10-14

Research and recovery notebooks both saved version 1 COMPLETE. The one F01 run
has exactly 10,000 authoritative and known physical journal updates, 40 complete
full sliding validations, 40 audited resumable checkpoint states and 41 native
immutable snapshots. No research failure, resume or replay was recorded.
Best update 1,750: loss 3.13024499, BPB 1.98376817, perplexity 22.879584.
Final update 10,000: loss 4.81814590, BPB 3.05346210, perplexity 123.735460.
One reserved test assessment followed selection freeze: BPB 2.24333983.

Actual training 4,269.206 s; validation 945.441 s; measured wrapper/archive
components sum 6,317.442 s, excluding queue/provisioning/publication/retrieval.
Archive 2,648,712,125 bytes, SHA256
`96de06da78ecb1416c8029eaf792003991a45a3dfa160eade43157bbe6ed78ab`.
Independent download took 11,213.970 s with retained partial range attempts;
complete CRC/safe paths/82-file inventory/all hashes, all checkpoint counters,
sampler replay and CPU inference passed in a separate 205.529 s inspection.
Recovered best/final CPU and CUDA greedy probes match exactly. Fresh qualified
CUDA restores all six full-state fields at 10,000 with zero additional updates,
verifies every native snapshot and reproduces all 100 fixed generation strings.

All 100 best/final and 50 internal 5k outputs remain unchanged. Strict ByteLevel
reconstruction found five invalid-UTF8 temperature-1.2 outputs, preserved without
repair. Late validation deterioration, greedy loops, malformed words and literal
training-passage copying are documented as negative findings. Engineering
acceptance does not establish semantic coherence or statistical superiority.

See report/final/F01_10000_analysis.md, interpretation.md, acceptance_checklist.md,
artifact_retrieval.md and evidence/research. PR19 remains unmerged for review.
The frozen F01 config's AUTHORIZED_NOT_TRAINED field records its creation-time
authorization; changing it would alter provenance. Current acceptance is recorded
in immutable execution receipts and these status documents. Historical pilot
definitions/results and all scientific source files remain unchanged.

Operational limitations preserved: initial 14-config-only test assertion,
independent float32 reference tolerance, local monitoring/test connection stalls,
read-only log reconnects, long retried archive transfer, and an existing-report
regeneration blocked by the no-replace writer. The latter was resolved using a
fresh staging directory and hash-preserving report promotion. None caused an
extra F01 research update or a scientific model modification.
