# E14 development persistence preflight — passed

This is development qualification, not E14 research results. Research is separately
submitted as one seed42/5000-update job and has no observed validation yet.

Frozen source: `ab1da4e05005413807d20802fff4153d97ebfa10`; scientific hash
`cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6` unchanged.
E13 PR16 is merged at2d698a671ebdae8e43d66e3062df174edd81cab0. The E14 model
passed20 CPU/CUDA width checks:240wide,6heads×40,MLP960,4blocks,context128,
sinusoidal/GELU/dropout0.1,2808960parameters. Qualified Py3.11.14/torch2.5.1+cu124/
CUDA12.4/NumPy1.26.4/full freeze and canonical dataset/tokenizer/manifest matched.
Actual Kaggle suite70passed in33.40s; final local69passed1CUDA-only skip116.89s.
SourceCI38409926632 and report-rendererCI38411947768 completed successfully.

## Actual saved outputs and read-back

- Development saved version1: https://www.kaggle.com/code/rayyannmerchant/nano-e14-development-20261013-fa3959be
- Fresh zero-update saved version1: https://www.kaggle.com/code/rayyannmerchant/nano-e14-development-recovery-20261013-c25df910
- Development archive282780282bytes; SHA256
  `981ddb740545097312b82660e9b3d521a3e03ae3352983aaa2710e2b61239679`.
- Independent Windows download588.585s; ZIPCRC, exact81-file inventory, every
  SHA256 and immutable smoke snapshot passed. Recovered CPU inference passed.
- Fresh compatible CUDA runtime independently verified the same81files and
  snapshot, restored model/optimizer/scheduler/scaler/Python-NumPy-torchCPU/CUDA-
  RNG/sampling/history exactly,1→1zeroextra updates, and performed inference.
- CPU/CUDA recovered checkpoint SHA and four-character continuation agree.
- All receipts are in `evidence/e14_development/`; the large downloaded archive,
  extracted states and transfer parts remain outsideGit on D:.

## Measured short benchmark and resource planning

Separate10-update benchmark, same4096targets/microbatch4/accumulation8/float32.
Excluding two warmup updates:5.321165updates/sec,21795.493targets/sec. Measured
full-validation23.818s, training1.887s, checkpoint34818570bytes, peak162896896
allocated GPU bytes. These are development measurements, not the5000-step result.
Projected5000 training-only939.644s; training+20validations+600s margin2016.011s,
under the conservative configured7200s envelope. This is an estimate, not a
verified end-to-end run time or a direct query of Kaggle's service timeout.

Actual immutable export+verification34826936bytes/0.575439s; benchmark exports
0.460860s and0.463704s. Aggregate archive construction+CRC16.104s. Storage check
required5629206248freebytes (100checkpoint-equivalents+2GiB) versus20801142784
available. Previous scientific jobs were complete; no E12 work/credentials or
shared artifact paths were used. Account quota before research:24.23GPUhours.

## Commands and operational boundaries

`build_notebook.py --experiment E14 --development-only --commit ab1da4e...`;
Kaggle push; independent `download_kaggle_archive.py --ranges`;
`inspect_development_archive.py ARCHIVE MANIFEST FRESH_OUTPUT`;
`build_notebook.py --development-recovery --source-kernel DEVELOPMENT/1`;
Kaggle push and genuine receipt retrieval. Research builder mounts that recovery
saved version; run.py verifies its exact source/state receipt before research.

The original development job queued, then ran and completed. Read-only log API
HTTP500/connection failures were retained separately and did not imply model
failure. Windows temporary-directory and initial CPU dataset-path errors were
corrected; the successful actual tests/CPU smoke are separately recorded.
No duplicate research job, unapproved seed/configuration or extension was launched.
The detailed E14 experiment remains pending until5000updates,20validations,
published research artifacts, independent CPU/CUDA recovery and report completion.
