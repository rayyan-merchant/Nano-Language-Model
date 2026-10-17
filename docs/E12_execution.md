> Historical execution workflow. Commands and notebook paths refer to the
> pinned trained source in the corresponding [research report](../report/README.md),
> not necessarily the current submission tree. See [historical retrieval](maintenance/README.md).

> Historical execution procedure. Current reports record experiment acceptance; individual earlier attempts may have failed. This is not renewed training authorization. Use [current reports and status](README.md) for actual acceptance and artifact retrieval. Pinned commands/source versions remain preserved below.

# E12 seed42 execution and delivery

E12 completed its sole authorized scientific run on 2026-10-13. Context 256 is the
only model override; 1,804,800 parameters, microbatch four, accumulation four,
float32 CUDA and 4,096 targets/update. Exactly 5,000 authoritative and known
physical research updates, twenty full validations, twenty checkpoint states and
twenty-one immutable snapshots are preserved. No research interruption or replay
occurred. No additional seed, configuration, combined run or extension was trained.

Reviewed main was synchronized at 2d698a6. Scientific code remains unchanged:
`cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6`.
Clean trained source: `e1f0c9861fcdd38d07cd24e746900d53a314b740`.
The separate E12 branch isolates orchestration/reporting from the primary
researcher's E11 work. No primary chat was available on this host; the reviewable
PR is the coordination boundary. Do not merge without approval.

Use the [complete report](../report/pilots/E12_seed42/E12_seed42_analysis.md),
[interpretation](../report/pilots/E12_seed42/interpretation.md) and its evidence.
Best/final BPB is 2.25627886 at 5,000, worse than matched E01 at all twenty points.
Training took 475.030 seconds, validation 927.393 seconds, peak allocation
277.908 MiB. The report records all metrics, comparisons, generation and limits.

## Persistent qualification and final recovery

Qualification version 358073577 passed all 67 CUDA tests in isolated Python
3.11.14/PyTorch 2.5.1+cu124/CUDA 12.4 on Tesla T4. It measured real E12 resource
and IO costs and published a real immutable smoke archive. Independent download,
CRC, safe paths and all file hashes passed. The initial notebook-output mount
stalled; a byte-identical compact private input was published. Unused stalled
E12 GPU draft was stopped on the same account. Fresh full-state roundtrip and
step-2→4 continuation were exact before substantive training. Receipt quick
version 358083761 preserved this gate. All development calls are separately labeled.

[Research saved version 358085129](https://www.kaggle.com/code/syedarijaali/e12-seed42-context-256-verified-research?scriptVersionId=358085129)
finished successfully. Complete archive: 407,902,560 bytes; SHA256
`d5f5c534cf3a8f939bbc52657fbb086b8b39ef28b42c52822bcc88153aea94a1`.
Seven published parts are at most 64 MiB. Independent complete-archive retrieval
passed all 171 hashes, ZIP CRC, paths, checkpoint counters/history, twenty full
validations, twenty-one reconstructed snapshots and CPU best/final inference.
Local slices match the independently retrieved published part index; separate
network downloads of every part are not claimed.

[Fresh CUDA recovery version 358138796](https://www.kaggle.com/code/syedarijaali/e12-seed42-final-cuda-recovery?scriptVersionId=358138796)
mounted published output in a new T4 job and verified full model, optimizer,
scheduler, scaler, RNG, sampling and history equality at 5000→5000, with zero
additional updates. Compatible software/freeze and source matched. Best/final
inference matched CPU recovery. Its saved receipt was independently downloaded
and agreed with published stdout. Both saved batch jobs finished; unused drafts
remain off. Working-directory files alone were not accepted as persistence.

## Reproduction and authorization boundaries

`build_e12_notebook.py` supports only preflight, research and recover-final.
Research first verifies durable smoke continuation, then starts only E12 seed42
and stops at 5,000. Recovery selects the recorded complete E12 lineage and runs
no training. Historical executed captures must not be blindly rerun; see
[notebook provenance](https://github.com/rayyan-merchant/Nano-Language-Model/blob/8f88edee89dfceb6d0a8c84e385a886c4e058356/notebooks/E12_notebook_provenance.md).

`verify_e12_archive.py` validates published part/file hashes, safe paths, checkpoint
counters/history, all snapshots, complete generation grid and CPU inference.
`write_e12_report.py` derives metrics and figures from verified checkpoint history,
checks reviewed E01/E11 environment/dataset/protocol equality and replays private
sampling without optimizer updates. Final reporting accepts an independently
retrieved zero-update CUDA receipt. The report preserves genuine negative outcomes.

`e12_resume.py` is available only for an incomplete hash-verified single E12
lineage and retains original attempted journals separately. It was not used in this
run. All large checkpoints, raw datasets, archives and expiring URLs stay outside
GitHub. PR #18 is for review; approval to merge or conduct further research is separate.
