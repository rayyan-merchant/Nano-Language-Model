# Recorded results and reproducible analysis

Current status: E01–E14 seed 42, F01 and B01 have completed, with genuine recorded metrics and recovery evidence. The original three-seed study was discontinued; missing replicates stay missing and sample standard deviation is undefined for one seed. This page replaces the obsolete bootstrap “NOT RUN” status and consolidates the former experiments/analysis/README guidance.

- [All fourteen pilot comparisons](../report/final/E01_E14_consolidated_analysis.md).
- [F01 measured analysis](../report/final/F01_10000_analysis.md) and [interpretation](../report/final/interpretation.md).
- [B01 measured analysis](../report/balanced/B01_seed42_analysis.md), [model selection](../report/balanced/B01_model_selection.md) and [matched E02/F01/B01 table](../report/balanced/E02_F01_B01_comparison.csv).
- [Complete report catalog](../report/README.md).

`nano_lm.results` is the strict shared pilot collector. It validates schemas, canonical variants, completed checkpoint/history identities, dataset/tokenizer and code/framework/evaluation policies; competing trajectories for the same experiment/seed are rejected, and identical copied runs are deduplicated. Its historical default expects the original 42-run grid. For the completed single-seed pilots, use `--allow-incomplete` with explicitly labeled seed completeness; do not alter the immutable collector to pretend the full grid exists.

F01/B01 have distinct configurations, aggregate fingerprints and budgets. Their accepted combined-model tables come from dedicated report utilities and genuine summaries; do not force incompatible runs into the pilot collector. Source code and supported reproduction entry points are cataloged in [scripts/README.md](../scripts/README.md).

Every conclusion should distinguish hypothesis, recorded observation, interpretation and limitations. Compare tokenizer-comparable BPB, parameter counts, genuine resource measurements and complete fixed generations; retain negative results, failure/replay accounting and measurement scopes. Never fill missing values with synthetic smoke metrics, interpolated validations or runtime projections presented as measurements. No additional training or original-test assessment is authorized by these analysis instructions.
