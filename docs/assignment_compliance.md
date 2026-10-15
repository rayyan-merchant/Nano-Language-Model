# Current assignment compliance and historical initialization audit

Research pipeline, genuine Tiny Shakespeare training/validation, conditional BPB, all14 single-seed ablations, F01/B01 inference/generations, measured resource plots and independent recovery are complete. Existing F01 architecture/RoPE material is [indexed here](../report/final/presentation_support/README.md).

Selected-model safetensors packaging, genuine FastAPI/frontend and public Vercel deployment are implemented. [Submission reproducibility and Colab](submission/README.md) are this phase and have separate execution gates. The final academic/Medium report, recorded presentation and comprehensive handout audit remain pending. The relocated [handwritten assignment PDF](handwritten-assignment.pdf) is preserved byte-for-byte; this cleanup does not certify its calculation correctness. Handout-specific manual three-token derivation/heatmaps, MLM-versus-CLM presentation and trained-attention evidence must be checked for the final submission, rather than inferred from model unit tests.

The original bootstrap audit below is preserved as historical context; its pending training/ablation statements have been superseded by accepted research. No final-submission-complete claim is made.

---

# Assignment compliance and initialization audit

Source: supplied A1_AIE.pdf at Downloads/7th sem/ai eng/A1, all 11 pages read.
No handout copy is added to the public repository. The existing handwritten PDF
is retained without claiming it has been validated.

Handout M1â€“M3: explicit Q/K/V attention, mask, multi-head split/merge, sinusoidal
positions and decoder block code exist; numerical, mask, shape and gradient tests
are implemented. Report-ready manual 3-token dk=dv=2 calculation, heatmaps,
architecture diagram and MLM/CLM comparison are pending artifacts.
M4: synthetic tiny-batch overfit and CPU smoke/resume tests implemented; genuine
Tiny Shakespeare training pipeline and validation logging remain pending.
M5â€“M6: full-corpus training, measured hardware/time/loss, five fixed-prompt
samples, trained attention visualization, ablation evidence, FastAPI hosting,
Medium report (recommended 8â€“12 pages), Colab demo and recorded presentation
remain pending; no final-submission claim is made.

Section 12 lists ten evidence categories despite the summary saying at least
six. Plan all ten plus the architecture diagram and objective comparison.
Section 16 requires clean setup, unit/overfit tests, final checkpoint reload,
three fixed prompts, future-token exclusion and README/security verification.
These final acceptance checks must use the final trained artifact, not smoke.
Section 15 requires at least two failure modes/limitations in the final report.

Master prompt Task A/B: inspect workspace/tools, verify identity and permission,
inspect remote, assign owner role. Task C: foundation files, environment, configs,
registry, docs, baseline modules, tests, notebook and CI; preserve prior remote
files. Task D is collaborator-only. Task E covers checkout, installation/tests,
configuration and forward shapes, plus branch workflow. Task F must report actual
commands/results, commit, missing capabilities and next steps. Advanced features
in sections 2â€“16 are project requirements, not evidence of bootstrap completion.
No 42-run study or long GPU jobs are authorized in initialization.

## Unified-pipeline update

Real-data acquisition/tokenization, training/validation, conditional BPB,
checkpoint restoration, inference, analysis utilities and platform smoke entry
points are implemented and audited in EXPERIMENT_READINESS.md. This does not
complete M5/M6: actual main ablations, final trained evidence, report, hosted
FastAPI service, recorded presentation and final acceptance remain pending.
