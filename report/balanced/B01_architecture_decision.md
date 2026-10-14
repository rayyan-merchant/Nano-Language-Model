# B01 architecture and stopping decision — before research

Reviewed main: d43cfbf7e3af6c982a8bcf1509a5f45b8e17ef4b, merged F01 PR19.
One fresh seed42 combined trajectory, at most5,000 updates; no original test
likelihood or additional dataset acquisition/evaluation. All previous report
files and configurations were read and remain unchanged.

| Experiment | Original factor | Parameters | Recorded best BPB | Best update |
|---|---|---:|---:|---:|
| E01 | baseline | 1,804,800 | 2.21405793 | 4750 |
| E02 | tokenizer | 2,173,056 | 2.02659521 | 5000 |
| E03 | position | 1,804,800 | 2.07327076 | 5000 |
| E04 | position | 1,829,376 | 2.23778869 | 5000 |
| E05 | activation | 1,804,800 | 2.24266813 | 5000 |
| E06 | heads | 1,804,800 | 2.20537989 | 4750 |
| E07 | heads | 1,804,800 | 2.21051529 | 5000 |
| E08 | depth | 2,694,528 | 2.14548940 | 4750 |
| E09 | depth | 3,584,256 | 2.10014737 | 4750 |
| E10 | mlp_expansion | 1,213,440 | 2.26711868 | 5000 |
| E11 | context | 1,804,800 | 2.15396251 | 5000 |
| E12 | context | 1,804,800 | 2.25627886 | 5000 |
| E13 | width | 1,021,824 | 2.32495848 | 5000 |
| E14 | width | 2,808,960 | 2.14189208 | 4750 |

F01's 6,046,560-parameter early minimum1.98376817 at1,750 is the proper quality
reference; its final3.05346210 is not used to manufacture a favorable B01 claim.
E02's compact2,173,056-parameter minimum2.02659521 remains the efficiency
reference. F01 training/validation divergence and long literal training passages
motivate attention to generalization; capacity, repeated exposure, constant LR,
optimization, corpus distribution and component interactions remain hypotheses,
not isolated causes. Gradient clipping is not established as causal.

The user locked B01 to canonical BPE1024, RoPE, six independently allocated
blocks, width192, six head32, MLP192→768→192/GELU, context128, preLN/dropout.1,
untied output. Count3,062,784: embeddings196,608 + sixblocks2,669,184 + LN384 +
head196,608. This is 49.3467% smaller than F01 and40.9436% larger than E02.
The additional capacity over E02 requires measured quality/efficiency evidence;
no additive forecast of pilot gains is made.

Stopping is predefined full sliding BPB every250, initialized at250; meaningful
reset iff bpb < reference−0.001, patience6, eligibility2,000, max5,000. Absolute
minimum selection has no tolerance. Planned250-step segments use the same run
and canonical exact-resume engine. Early stopping is normal termination, not a
failure. It changes selection/budget and cannot guarantee coherent text.

Preserve every observed point; do not invent the unexecuted remainder. Compare
E02/F01/B01 fixed100-slot outputs, source-byte exposure, best/final BPB, actual
resources and clipping. Same tokenizer/context aligns the comparison better
than character models, but capacity/depth/positions and initialization differ;
one seed and repeated validation use supply no significance estimate.

Canonical BPE and character artifacts are copied from the audited prepared
pipeline into assets/tokenizers for exact clean-checkout reuse. B01 preparation
acquires only the pinned Tiny Shakespeare source, uses shared splitting and
validation, and never fits either vocabulary. Artifact hashes and manifests
must remain identical. No scientific src module change is planned.
