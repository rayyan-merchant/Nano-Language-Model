> Historical execution workflow. Commands and notebook paths refer to the
> pinned trained source in the corresponding [research report](../report/README.md),
> not necessarily the current submission tree. See [historical retrieval](maintenance/README.md).

> Historical execution procedure. Current reports record experiment acceptance; individual earlier attempts may have failed. This is not renewed training authorization. Use [current reports and status](README.md) for actual acceptance and artifact retrieval. Pinned commands/source versions remain preserved below.

# E14 execution and acceptance record


## E14 width240 seed42 — accepted 2026-10-14

E13 PR16 was verified merged before preparation. Frozen trained source
`ab1da4e05005413807d20802fff4153d97ebfa10`; scientific hash
`cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6` unchanged.
Only width240, deriving head40/MLP960; 2,808,960 parameters (+55.638%).
A separate published E14 development artifact passed independent 81-file/CRC/
snapshot/CPU/fresh-CUDA full-state recovery before research. The one authorized
run completed exactly 5,000 authoritative/known physical updates and 20 full
sliding validations, context128/microbatch4/accumulation8/4,096 targets.
Best4750 loss1.48464646/BPB2.14189208; final loss1.48683957/BPB2.14505608.
Best BPB is3.259% lower than E01; training978.158s (+6.703%), validation490.769s,
recorded allocator peak177.283MiB. Source-clone to archive1903.251s.

Research archive623,713,697bytes, SHA256
`e41393c4a97c2ec565558df4ca0997d74275cf62e69c57b466c4794855b26fd7` independently
passed CRC/safe paths/exact62-file inventory/allhashes, all20optimizer/scheduler
counters, sampler replay and best/final CPU inference. All21published snapshot
manifests matched; fresh compatible CUDA final full-state5000→5000zeroextra and
best/final inference matched. All100 unedited generations, 20PNG/20PDF figures,
full E13/E01/E14 comparisons and detailed report are preserved. No research
failure/resume/replay; log-service disconnects retained separately. CPU69passed
1CUDA-only skip; actual CUDA70passed. Single-seed/confounding/peak-sampling scopes
are explicit. No other research run or extension launched; study-wide approval
remains separately gated. See report/pilots/E14_seed42/E14_seed42_analysis.md.

Research saved version1: rayyannmerchant/nano-e14-pilot-20261013-82e48728.
Fresh zero-update saved version1: rayyannmerchant/nano-e14-recovery-20261013-1bd9f987.
Qualified T4GPU0/Python3.11.14/torch2.5.1+cu124/CUDA12.4/NumPy1.26.4/full freeze matched.
Canonical raw/tokenizer/partition/manifest identities matched; 55,514 validation
bytes scored after256 conditioning-only bytes. Source and reporting commits are
separate; scientific modules/configurations and earlier experiment results remain
unchanged. Open E14 PR for review; do not merge automatically.
