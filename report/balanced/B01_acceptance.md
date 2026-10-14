# B01 execution and independent recovery acceptance

These are genuine B01 seed42 measurements and receipts. Engineering acceptance
does not establish model superiority, statistical significance or general text
competence. This is one separately approved combined architecture; the former
three-seed grid remains discontinued and no subsequent training is authorized.

| Requirement | Actual evidence |
|---|---|
| Frozen architecture | BPE1024, width192, six independent blocks, six heads32, RoPE Q/K, context128, MLP768/GELU/preLN/dropout0.1; exactly3,062,784 parameters; CPU and qualified CUDA numerical receipts |
| Scientific implementation unchanged | Normalized source fingerprint d5b205aecec5913cd11d921c0a84899badd95a014a490a7beaab3d2598c940f8; historical src/configs/reports diff is empty |
| Clean committed training source | c2cf1c165d09e0fc8f64a2df714755a4d910d469; clean checkout recorded in published environment; stopping rule committed before launch |
| Canonical data/tokenizer | Pinned raw/BPE/manifest/split identities verified; original audited artifacts reused without fitting; train-only vocabulary lineage preserved |
| Qualified GPU | Tesla T4 device0, Python3.11.14, torch2.5.1+cu124/CUDA12.4, NumPy1.26.4 and exact dependency freeze; float32 |
| Deterministic segmentation | Development uninterrupted4 versus2+2: model/optimizer/scheduler/scaler/allRNG and every scientific training-history field equal; actual clocks/throughput/allocator telemetry excluded and disclosed |
| Raw complete-state roundtrip | All six fields equal including complete telemetry/history; published development fresh CUDA passed before research launch |
| One research trajectory | One Kaggle notebook version, one unique run ID, 17 planned boundary resumes; 4,500 authoritative/known physical research updates, no interruption replay in recorded journal |
| Frozen stop | At4500, patience reached six after best3000; first legitimate eligible stopping boundary; independent rule replay passes; no later optimizer row exists |
| Full evaluation | Eighteen actual validations250..4500; each55,514 original bytes/24,386 BPE targets, common256-byte conditioning exclusion, full sliding context128/dropout off; no fabricated4750/5000 observations |
| Complete preserved states | Eighteen resumable validation checkpoints; 37 native immutable snapshots (two per planned segment plus final stopping-summary snapshot); all counters/manifests verified |
| Correct checkpoint selection | Absolute best3000 BPB1.9849309521722194, independent of the0.001 patience threshold; final4500 BPB2.044457963659289 |
| Durable publication | Research saved Kaggle version1 COMPLETE; full archive independently downloaded from that published output |
| Independent archive integrity | 604,825,667bytes; exact96-file inventory, safe paths/no duplicates, ZIP CRC and every file SHA256 passed |
| Recovered CPU inference | Best/final models genuinely generate64 new tokens; both ROMEO greedy texts equal prefixes of the original CUDA fixed256-token outputs |
| Fresh compatible CUDA full state | Recovery4500→4500; model/optimizer/scheduler/scaler/allRNG/sampler/full training state exactly equal; zero additional optimizer updates |
| Fixed generation reproduced | All100 strings reproduced; independently compared original/recovered256 token IDs per slot; five invalidUTF8 outputs retained without repair |
| Test policy | No B01 original-test likelihood or external corpus evaluation/download; prior F01 test exposure acknowledged, not used to tune B01 |
| Delivery | Actual reports, full fixed grid, figures, matched comparison data and receipts on dedicated B01 branch and PR20; no automatic merge or large artifacts in Git |

## Published identities and integrity

- Research: https://www.kaggle.com/code/rayyannmerchant/nano-b01-research-20261014-33317a00 (version1).
- Recovery: https://www.kaggle.com/code/rayyannmerchant/nano-b01-research-recovery-20261014-d2c9b763 (version1).
- Run: `B01_seed42_c1b6b391cd9b476994325a85c264493a`.
- Resolved configuration hash: `ec95f5c5d24a950b81ffe1b20dbbfc1ed95f9134965b77897e21924412228125`.
- Aggregate source/configuration hash: `4dd230985194865487254e2e750b07f79aeabec491e4fa8f8e483fedda1de741`.
- Archive SHA256: `41742da357777a6aee40bd7bc4a4063ee1e3d07f94b94e2344f082252a467edf`.
- PR: https://github.com/rayyan-merchant/Nano-Language-Model/pull/20.

The original pre-F01 aggregatecdea42... changes when approved combined config
definitions are added. This is not a change in frozen scientific Python source;
normalized source fingerprint and historical git diff establish that distinction.

## Failures and scope limitations

The first new CPU test used implicit Windows cp1252 and failed reading the audited
BPE artifact. Explicit UTF8 corrected the reader. A separate local archive
filename helper failed before any transfer due to an eager legacy fallback;
fixed/tested without changing GPU execution. Read-only Kaggle log streams
occasionally disconnected; reconnects did not submit or restart training.
During local genuine-grid display, cp1252 could not print a replacement character;
stdoutUTF8 corrected display without changing generated data. Local Matplotlib
dependency deprecation warnings and native notebook-renderer syntax warnings
are recorded; headless scientific plots and notebook publication succeeded.

Exact trajectory equivalence excludes measured clocks and allocator telemetry;
raw checkpoint restoration excludes nothing. Training peak memory resets at each
segment boundary and is sampled before validation; separate validation/inference
peaks were not measured. F01 uninterrupted peaks can include earlier validation,
so memory comparisons have a scope confound. Inference timing is synchronized
generation only, excluding tokenize/decode/model load/CUDA transfer; CPU loading
is cache-dependent. Comparable E02/F01 request latency is unavailable.

All models have one seed, differing parameter counts, initialization, depth,
positions, budgets and selection opportunities. B01's best likelihood does not
establish better grammar: greedy loops, malformed labels/words, scene drift and
invalid byte sequences remain. Exact training phrase overlap is an indicator,
not proof of a unique memorization mechanism. B01's unexecuted future trajectory
is unknown; early stopping is a frozen heuristic, not a proven optimal budget.
