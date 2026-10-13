# Durable E13 artifact retrieval

Research savedversion1:
https://www.kaggle.com/code/rayyannmerchant/nano-e13-pilot-20261013-ee39c7d3

FreshCUDA recovery savedversion1:
https://www.kaggle.com/code/rayyannmerchant/nano-e13-recovery-20261013-a50107ba

Download E13_seed42_pilot.zip and artifact_manifest.json from the research output.
ZIP230564978bytes; SHA2560ddce4c76829e961ff862ab61f7d67f9a8ca679260d4fc3a8219fc617becf7e7.
Archivecontainsall20resumablecheckpointstates,completejournals/config/tokenizer/
data/environment,100generations,figures andnative21-snapshot inventory. Duplicate
immutable snapshotweightfiles remain in savedoutputs underpilot/snapshots.

Independently verified local archive:
D:/Nano-Language-Model-artifacts/E13_published_20261013_ee39c7d3/E13_seed42_pilot.zip

Extracted verified run:
D:/Nano-Language-Model-artifacts/E13_published_20261013_ee39c7d3/verified/runs/E13_seed42_40d4850f3eb143058a6a447e071252c6

Best=finalcheckpoints/step_005000.pt SHA256fad206f744375ae0bbe9618017ec87ad16b50c8377957a449c13c8fffea10896.

Use scripts/pilot/download_kaggle_archive.py --ranges with the legitimate account,
publishedversion1 andexactmanifest; signedURLs/credentials are never recorded.
Use inspect_archive.py ARCHIVE MANIFEST EXCLUSIVE_NEW_OUTPUT for SHA256/size/
CRC/path/fileinventory/hash/history/bestfinalCPU checks. FreshCUDA restoration
uses build_notebook.py --experiment E13 --restore-only --source-kernel
rayyannmerchant/nano-e13-pilot-20261013-ee39c7d3/1 --commit
a62cd6cebda71d5fbe1606add9efd9f71171076a, which never invokes the training CLI.
Do not rerun already completed recovery merely to inspect itsreceipt.
No10kextension is authorized. Later continuation requires separate approval and
exact compatible optimizer/scheduler/scaler/RNG/sampler/history continuity.
