> Historical execution workflow. Commands and notebook paths refer to the
> pinned trained source in the corresponding [research report](../report/README.md),
> not necessarily the current submission tree. See [historical retrieval](maintenance/README.md).

> Historical execution procedure. Current reports record experiment acceptance; individual earlier attempts may have failed. This is not renewed training authorization. Use [current reports and status](README.md) for actual acceptance and artifact retrieval. Pinned commands/source versions remain preserved below.

# E13 execution and acceptance record

## E13 seed42 width144 — accepted 2026-10-13

E11exactfreshCUDAacceptance and PR15integration were reverified beforeE13. Clean
trainedsourcea62cd6cebda71d5fbe1606add9efd9f71171076a;scientific hashcdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6unchanged. Onlywidth144/head24/MLP576 mechanicallyderived;1021824parameters(−43.383%). Exactly5000updates/20fullvalidations/context128/microbatch4/accumulation8/4096targets. Best=final5000loss1.61153841/BPB2.32495848,5.009%higherBPBthanE01-best. Training941.393s,validation310.326s,pipeline1625.390s,peak132.291MiB. No recordedresearchfailure/resume;no replay.

Published230564978-byte archiveSHA2560ddce4c76829e961ff862ab61f7d67f9a8ca679260d4fc3a8219fc617becf7e7independently passedCRC/exactinventory/all61hashes/CPUbestfinalinference/all20counters/samplerreplay/21snapshotmanifest parity. FreshqualifiedCUDAallstate5000to5000zeroextra andCPU/CUDAgeneration match passed. Full100uneditedgrid/15PNG/15PDFfigures/machinecomparisons/detailedreport preserved outsideGitforlargefiles. CPU63passed1skip;actualGPU64passed. NoE12/E14/otherseed/hybrid/extension. E13accepted;study-wideapprovalremains separatelygated. See report/pilots/E13_seed42/E13_seed42_analysis.md.

Researchversion1:rayyannmerchant/nano-e13-pilot-20261013-ee39c7d3. Freshzero-updaterecoveryversion1:rayyannmerchant/nano-e13-recovery-20261013-a50107ba. QualifiedT4GPU0/Python3.11.14/torch2.5.1+cu124/CUDA12.4/NumPy1.26.4/freeze matched. Canonicalraw/tokenizer/partitionhashes matched;55514validationbytes scoredafter256conditioningbytes. ActualCUDA20-check widthaudit,fullrealvalidation,checkpoint/snapshot/allstateroundtrip/Agg/ZIP/generation passedbeforefreshresearch reseeding. No testlikelihood. Sourcechangeswere thinorchestration/tests/reporting;modelmathematics unchanged. Do not mergePR16withoutreview.
