# Project map and current status

NanoLM Studio implementation is documented in [the local/deployment guide](deployment/README.md)
and [its verification record](deployment/verification.md). Genuine E02/B01/F01 inference
is locally verified using the exact selected checkpoints. The user authorized the
[Vercel-only application rollout](deployment/hosting.md); additional research execution remains unauthorized.

Current reviewed research baseline: `cc7e345cee92c122cb1f7ad56a0d7627636934db`,
including merged B01 PR20. **Research execution is complete; final project delivery
is not.** E01–E14 each have one accepted seed 42/5,000-update authoritative
trajectory and20 validations. F01 completed10,000/40; B01's frozen rule stopped
at4,500/18. The proposed three-seed grid was discontinued. Engineering acceptance
and preservation do not imply significance, coherent language or an optimal model.

## Where to start

| Purpose | Read |
|---|---|
| Setup, scope and commands | [root README](../README.md), [commands](commands.md), [script catalog](../scripts/README.md) |
| Architecture/source | [explicit attention](../src/nano_lm/attention.py), [model](../src/nano_lm/model.py), [positions](../src/nano_lm/positions.py), [numerical tests](../tests) |
| Locked configurations | [baseline](../configs/baseline.json), [registry](../experiments/registry.csv), [sixteen configurations](../experiments/configs) |
| Method and limitations | [research protocol](research_protocol.md), [original experiment design](experiment_design.md), [BPB](bpb.md) |
| Data/vocabulary provenance | [dataset guidance](../data/README.md), [canonical tokenizer artifacts](../assets/tokenizers), [recorded hashes](../EXPERIMENT_READINESS.md) |
| Platforms/persistence | [platforms](platforms.md), [storage](storage.md), [qualified Kaggle environment](qualifications/KAGGLE_RAYYAN_20261011.md) |
| Completed notebook records | [notebook catalog](../notebooks/README.md) |
| Current conclusions | [recorded analysis](results_analysis.md), [report catalog](../report/README.md) |
| Historical chronology | [progress](progress.md), [decisions](decisions.md) — dated claims retain their original scope |
| Remaining assignment work | [compliance](assignment_compliance.md), [preserved handwritten artifact](handwritten-assignment.pdf) |
| Repository maintenance | [inventory](maintenance/repository_inventory.csv), [plan](maintenance/cleanup_plan.md), [changelog](maintenance/cleanup_changelog.md), [verification](maintenance/cleanup_verification.md) |

The frozen F01 presentation findings guide has two historical relative href
defects. Use these correct direct destinations: [F01 analysis](../report/final/F01_10000_analysis.md)
and [complete generation grid](../report/final/generation.md). Its original bytes
remain protected by the unchanged B01 evidence-parity guard.

## All fourteen seed 42 pilot reports

| ID | Intended factor | Genuine research report |
|---|---|---|
| E01 | Baseline | [E01](../report/pilots/E01_seed42/E01_seed42_analysis.md) |
| E02 | BPE tokenizer | [E02](../report/pilots/E02_seed42/E02_seed42_analysis.md) |
| E03 | RoPE | [E03](../report/pilots/E03_seed42/E03_seed42_analysis.md) |
| E04 | Learned positions | [E04](../report/pilots/E04_seed42/E04_seed42_analysis.md) |
| E05 | ReLU | [E05](../report/pilots/E05_seed42/E05_seed42_analysis.md) |
| E06 | Four attention heads | [E06](../report/pilots/E06_seed42/E06_seed42_analysis.md) |
| E07 | Eight attention heads | [E07](../report/pilots/E07_seed42/E07_seed42_analysis.md) |
| E08 | Six blocks | [E08](../report/pilots/E08_seed42/E08_seed42_analysis.md) |
| E09 | Eight blocks | [E09](../report/pilots/E09_seed42/E09_seed42_analysis.md) |
| E10 | MLP expansion2× | [E10](../report/pilots/E10_seed42/E10_seed42_analysis.md) |
| E11 | Context64 | [E11](../report/pilots/E11_seed42/E11_seed42_analysis.md) |
| E12 | Context256 | [E12](../report/pilots/E12_seed42/E12_seed42_analysis.md) |
| E13 | Width144 | [E13](../report/pilots/E13_seed42/E13_seed42_analysis.md) |
| E14 | Width240 | [E14](../report/pilots/E14_seed42/E14_seed42_analysis.md) |

These count authoritative trajectories, not necessarily physical attempted
updates after interruptions; each affected report retains replay/failure accounting.
Source commits, account/platform differences and exact recovery receipts are in
the individual reports, not flattened into an invented uniform environment.

## Combined models and trusted artifacts

- F01: [analysis](../report/final/F01_10000_analysis.md),
  [acceptance](../report/final/acceptance_checklist.md),
  [best1,750/final10,000 retrieval](../report/final/artifact_retrieval.md).
- B01: [analysis](../report/balanced/B01_seed42_analysis.md),
  [model selection](../report/balanced/B01_model_selection.md),
  [acceptance](../report/balanced/B01_acceptance.md),
  [best3,000/final4,500 retrieval](../report/balanced/artifact_retrieval.md).
- E02's best/final5,000 BPE state and all other pilot checkpoints are identified
  by their report's actual archive manifest, published version and retrieval record.

Large model states, datasets, ZIPs and full journals are external. Cleanup preserves
their exact instructions/hashes; it does not re-download or certify new platform
availability. Never deserialize an untrusted checkpoint or rerun research notebooks
for convenience. F01's reserved test was assessed once; B01 did not use that test.

## Subsequent milestones — not implemented by cleanup

Final academic/Medium report and handout-specific evidence audit; selected-model
inference packaging; final reproducible demo notebook; FastAPI under [api](../api/README.md);
frontend, deployment and recorded presentation. Existing source inference and
research reports are available, but none proves those delivery milestones complete.
No additional training/test assessment, new dependency policy or automatic PR merge
is authorized. [AGENTS.md](../AGENTS.md) protects scientific paths and evidence.
