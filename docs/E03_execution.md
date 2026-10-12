# Authorized E03 seed42 execution

Rayyan explicitly authorized one E03/RoPE run, seed42, exactly5000 completed
optimizer updates on rayyannmerchant Kaggle. No other run or10k extension is
authorized. E02 belongs to the other member's independent Colab execution;
this branch does not touch it. Broader three-seed study approval remains separate.

Main synchronized to reviewed merge2cf39518492652e2ed05e459318e6166e4275814.
Scientific src/configs/experiment files are unchanged from trained E01 source
444bca4c4ac19f97f46fae210a9fd361f8561537. Expected core hash:
cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6.
The execution wrapper rejects any difference. Canonical E03 changes position only
to rope; width192/layers4/heads6/head dimension32/context128/MLP4/GELU/dropout.1,
character vocabulary65 and parameter count1804800 remain the same.

## Architecture audit

rope_audit.py independently checks scalar adjacent-coordinate 2x2 rotations,
pairwise norm preservation, autograd, projected/head-split Q/K-only rotation,
unrotated V aggregation, causal mask and dimensions, finite logits/loss, manual
cross entropy, absence of learned positions and unused sinusoidal buffer, future
token invariance, Q/K/V parameter gradients and cropped generation position reset.
Same seed creates exactly matching E01/E03 initial trainable tensors. It never
loads E01 weights for training. The audit runs both locally and on CUDA before
substantive training. It measures local correctness, not extrapolation quality.

## Orchestration review and gates

Changes are confined to scripts/pilot, tests and documentation. Existing E01
results remain byte-for-byte untouched. run.py continues to call the canonical
CLI; experiment becomes an explicit restricted E01/E03 argument. analyze.py
derives labels from checkpoint metadata. Development postprocessing accepts only
smoke metadata and actual capped validation; its outputs are separate and excluded
from the research archive. It tests all plotting functions and genuine generation
before research. Agg is set before imports. Fresh recovery-only notebooks invoke
recover.py, never run.py. Snapshot restore writes into a fresh writable output,
leaving mounted sources untouched. The original failure-recovery utility remains
an E01-specific rescue path, not the intended successful E03 restore workflow.

Review covers exact argument propagation, metadata-derived labels, no second
training loop, no model/config edits, reseeding after development checks, archive
selection and zero-update recovery. Local numerical tests, generated-notebook
compilation/argument tests, real-data CPU development postprocessing, full suite
and branch CI must pass before freezing the clean execution commit. This is
engineering code review/testing; no human PR approval is represented as obtained.

Research settings are exactly E01's approved pilot overrides: float32 CUDA,
seed42,AdamW3e-4 constant,weight decay.01,clip1,microbatch4/accumulation8,
4096 predicted targets/update,eval250,checkpoint500 with existing validation saves,
full sliding256 conditioning bytes and55514 scored bytes, no test likelihood.
Isolated Python3.11.14/torch2.5.1+cu124/CUDA12.4/NumPy1.26.4 and full qualified
transitive freeze are mandatory; native notebook packages remain untouched.

Every run/notebook/snapshot has a unique identity. Save the output version,
independently download/hash/CRC-check the complete archive, generate best/final
inference, and mount published output in a fresh CUDA runtime for exact state
roundtrip with zero further updates. Do not cancel the runtime before publication.
Full per-step history and all20 checkpoints stay in durable outputs, not GitHub.
Only small evidence/results/figures/report are committed. All comparisons against
E01 are one-seed descriptive comparisons, with matching byte/scoring policy and
hardware/runtime caveats. Training, validation, IO and complete pipeline scopes
remain separate. Actual result evidence is appended only after execution.


## E03 seed42 execution evidence — 2026-10-12

The individually authorized E03 RoPE run completed exactly5000 updates and20 full
sliding validations. Clean training source0dd75296a773081f55a7a6d1da111e482562c1c1;
scientific core hash matches E01. Best/final update5000: loss1.43708178,
conditional BPB2.07327076,1804800 parameters. Train1107.721s; validation447.515s;
complete measured notebook pipeline1951.871s; peakallocated151.71MiB.
Published original notebook82f33b8a version1 and fresh recovery33b2c3ba version1
are COMPLETE on rayyannmerchant. Archive402094081bytes SHA256
57feb9d606a1d733ce3ae301b9e401c12b200f32c1d3c3fca4c47c3c6117b110:
58 files independently verified;20 checkpoints/21 snapshots retained. CPU
inference and fresh CUDA full-state restoration/generation passed with zero
additional updates. Local C-drive extraction failure was recovered to unique D
storage without changing E01 evidence. Report/evidence:
report/pilots/E03_seed42/E03_seed42_analysis.md and interpretation.md.
Pre-execution Windows tests36 passed,1 CUDA skip; actual CUDA RoPE audit13 passed.
E03 improves final loss6.44% with20.84% more training time; greedy generation is
more repetitive on4/5 fixed prompts. Single-seed descriptive comparison only.
No other run or extension launched. E02 is the colleague's independent Colab
execution, outside this evidence. The broader three-seed study remains separately
gated; scoped E01/E03 completion is not global GO. Further experiments/extensions
require authorization. PR6 is reviewable and is not merged.
