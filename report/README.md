# Research index

Completed measured research: E01 baseline and thirteen single-factor variants at
seed 42, plus separately approved combined F01/B01. No multi-seed significance is
claimed. Reports distinguish authoritative trajectories from replayed physical
updates, hardware/session differences and genuine failures.

## Fourteen pilots

| Experiment | Interpretation and provenance | Metrics | All validation observations | Complete generation grid |
|---|---|---|---|---|
| E01 | [analysis](pilots/E01_seed42/E01_seed42_analysis.md) | [summary](pilots/E01_seed42/summary.json) | [validation](pilots/E01_seed42/validation.csv) | [generations](pilots/E01_seed42/generation.json) |
| E02 | [analysis](pilots/E02_seed42/E02_seed42_analysis.md) | [summary](pilots/E02_seed42/summary.json) | [validation](pilots/E02_seed42/validation.csv) | [generations](pilots/E02_seed42/generation.json) |
| E03 | [analysis](pilots/E03_seed42/E03_seed42_analysis.md) | [summary](pilots/E03_seed42/summary.json) | [validation](pilots/E03_seed42/validation.csv) | [generations](pilots/E03_seed42/generation.json) |
| E04 | [analysis](pilots/E04_seed42/E04_seed42_analysis.md) | [summary](pilots/E04_seed42/summary.json) | [validation](pilots/E04_seed42/validation.csv) | [generations](pilots/E04_seed42/generation.json) |
| E05 | [analysis](pilots/E05_seed42/E05_seed42_analysis.md) | [summary](pilots/E05_seed42/summary.json) | [validation](pilots/E05_seed42/validation.csv) | [generations](pilots/E05_seed42/generation.json) |
| E06 | [analysis](pilots/E06_seed42/E06_seed42_analysis.md) | [summary](pilots/E06_seed42/summary.json) | [validation](pilots/E06_seed42/validation.csv) | [generations](pilots/E06_seed42/generation.json) |
| E07 | [analysis](pilots/E07_seed42/E07_seed42_analysis.md) | [summary](pilots/E07_seed42/summary.json) | [validation](pilots/E07_seed42/validation.csv) | [generations](pilots/E07_seed42/generation.json) |
| E08 | [analysis](pilots/E08_seed42/E08_seed42_analysis.md) | [summary](pilots/E08_seed42/summary.json) | [validation](pilots/E08_seed42/validation.csv) | [generations](pilots/E08_seed42/generation.json) |
| E09 | [analysis](pilots/E09_seed42/E09_seed42_analysis.md) | [summary](pilots/E09_seed42/summary.json) | [validation](pilots/E09_seed42/validation.csv) | [generations](pilots/E09_seed42/generation.json) |
| E10 | [analysis](pilots/E10_seed42/E10_seed42_analysis.md) | [summary](pilots/E10_seed42/summary.json) | [validation](pilots/E10_seed42/validation.csv) | [generations](pilots/E10_seed42/generation.json) |
| E11 | [analysis](pilots/E11_seed42/E11_seed42_analysis.md) | [summary](pilots/E11_seed42/summary.json) | [validation](pilots/E11_seed42/validation.csv) | [generations](pilots/E11_seed42/generation.json) |
| E12 | [analysis](pilots/E12_seed42/E12_seed42_analysis.md) | [summary](pilots/E12_seed42/summary.json) | [validation](pilots/E12_seed42/validation.csv) | [generations](pilots/E12_seed42/generation.json) |
| E13 | [analysis](pilots/E13_seed42/E13_seed42_analysis.md) | [summary](pilots/E13_seed42/summary.json) | [validation](pilots/E13_seed42/validation.csv) | [generations](pilots/E13_seed42/generation.json) |
| E14 | [analysis](pilots/E14_seed42/E14_seed42_analysis.md) | [summary](pilots/E14_seed42/summary.json) | [validation](pilots/E14_seed42/validation.csv) | [generations](pilots/E14_seed42/generation.json) |

[Consolidated analysis](final/E01_E14_consolidated_analysis.md) ·
[Master comparison CSV](final/master_comparison.csv) ·
[All matched validation points](final/all_pilots_matched_validation.csv).
Selected loss/BPB plots remain in each pilot's figures directory.

## Combined models

| Model | Analysis and selection | Numerical inputs | Genuine generation | Large checkpoint retrieval |
|---|---|---|---|---|
| F01 | [complete report](final/F01_10000_analysis.md), [interpretation](final/interpretation.md), [acceptance](https://github.com/rayyan-merchant/Nano-Language-Model/blob/5f3bfd8b3a95c346be9f70aa0404e1cdf2fbd430/report/final/acceptance_checklist.md) | [summary](final/summary.json), [validation](final/validation.csv) | [grid](final/generation.json), [readable](https://github.com/rayyan-merchant/Nano-Language-Model/blob/5f3bfd8b3a95c346be9f70aa0404e1cdf2fbd430/report/final/generation.md) | [instructions](final/artifact_retrieval.md) |
| B01 | [complete report](balanced/B01_seed42_analysis.md), [selection](balanced/B01_model_selection.md), [acceptance](https://github.com/rayyan-merchant/Nano-Language-Model/blob/5f3bfd8b3a95c346be9f70aa0404e1cdf2fbd430/report/balanced/B01_acceptance.md) | [summary](balanced/summary.json), [validation](balanced/validation.csv), [comparison](balanced/E02_F01_B01_comparison.csv) | [grid](balanced/generation.json), [readable](https://github.com/rayyan-merchant/Nano-Language-Model/blob/5f3bfd8b3a95c346be9f70aa0404e1cdf2fbd430/report/balanced/generation.md) | [instructions](balanced/artifact_retrieval.md) |

The selected loss and BPB plots are in [F01 figures](final/figures) and
[B01 figures](balanced/figures). Artifact manifests remain under each report's evidence directory. Detailed
environment and independent recovery receipts are available through the
[pinned historical archive](../docs/maintenance/README.md).

## Reproduce and interpret

Scientific training uses the shared [nano_lm engine](../src/nano_lm), frozen
[configurations](../experiments/configs) and [dataset/tokenizer provenance](../data/README.md).
[Commands](../docs/commands.md) and [script interfaces](../scripts/README.md) cover
training, analysis, recovery and figure generation. Historical platform wrappers
must use their pinned source/dependencies; do not launch new research without
separate authorization.

[Submission figure reproduction](../tools/assignment_evidence.py) uses authentic
retained measurements and educational references. Use a scratch destination;
never overwrite accepted evidence. The [submission evidence index](../docs/submission/evidence/README.md)
provides the final required diagrams and executed inference/attention receipts.

Conditional BPB scores a common validation suffix with each configured context;
loss/perplexity across tokenizers are not directly comparable. Equal predicted
token budgets do not equalize byte exposure or computation. F01/B01 combine
factors and have different stopping opportunities; they are not controlled
single-factor ablations. All known generation failures remain visible.

## Historical material

Intermediate plots, duplicated PDF exports, provisioning notebooks and
preflight attempts were removed from the current tree. Their exact original
paths and Git blobs are mapped in [the preservation ledger](../docs/maintenance/public_tree.csv).
[Retrieval instructions](../docs/maintenance/README.md) access the original public
commit without rewriting history. Existing research narratives retain their
results and point archived links to that commit. Large datasets/model archives
remain externally hosted, not copied into Git.
