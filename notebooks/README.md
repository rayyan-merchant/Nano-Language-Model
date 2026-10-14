# Notebook catalog

All 15 notebook files and their provenance bytes/paths are preserved. Five other files in this directory are navigation/provenance records; the directory's20 files are not20 executable notebooks. None was executed by cleanup. These are thin wrappers around the pinned shared package, not separate training implementations.

## Current development entry points

| Notebook | Scope |
|---|---|
| [bootstrap.ipynb](bootstrap.ipynb) | Synthetic CPU foundation smoke; not trained-model evidence or the final submission demo |
| [kaggle_research.ipynb](kaggle_research.ipynb) | Shared-package short CUDA verification; requires compatible isolated profile and actual storage qualification |
| [colab_research.ipynb](colab_research.ipynb) | Shared-package short CUDA verification; not authorization for research training |

Native cloud Python/PyTorch may differ from the qualified profile. Read [platform guidance](../docs/platforms.md) before any separately approved execution. The final demo/submission notebook remains a later task.

## Source-only historical launchers

| Notebook | Preserved purpose |
|---|---|
| [colab_e02_seed42_pilot.ipynb](colab_e02_seed42_pilot.ipynb) | Original E02 BPE Colab provisioning/research procedure |
| [colab_e05_seed42_pilot.ipynb](colab_e05_seed42_pilot.ipynb) | Clean E05 launcher; research disabled by default |
| [colab_e07_seed42_pilot.ipynb](colab_e07_seed42_pilot.ipynb) | Clean E07 Colab launcher; distinct from the failed allocation snapshot |
| [kaggle_e07_seed42_qualification.ipynb](kaggle_e07_seed42_qualification.ipynb) | Historical E07 account/profile/persistence qualification |
| [kaggle_e07_seed42_research.ipynb](kaggle_e07_seed42_research.ipynb) | Historical E07 Kaggle research launcher |
| [kaggle_e12_seed42_research.ipynb](kaggle_e12_seed42_research.ipynb) | Historical E12 context256 launcher |

These contain exact source/account/protocol assumptions. Use their recorded trained commit when studying reproduction; do not change source guards or rerun completed trajectories on current main. Source-only files do not prove execution by themselves.

## Genuine executed and failed-attempt records

| Notebook | Evidence and scope |
|---|---|
| [E05_seed42_ReLU_Colab_pilot.ipynb](E05_seed42_ReLU_Colab_pilot.ipynb) | Ten-cell completed Colab capture with actual outputs and final evidence; [provenance](E05_seed42_ReLU_Colab_pilot.provenance.json) |
| [E07_seed42_EightHead_Colab_pilot.ipynb](E07_seed42_EightHead_Colab_pilot.ipynb) | One-cell failed T4 allocation attempt before training; no completed-run evidence; [provenance](E07_seed42_EightHead_Colab_pilot.provenance.json) |
| [kaggle_e07_seed42_research_executed.ipynb](kaggle_e07_seed42_research_executed.ipynb) | Actual Kaggle research capture; interruption/replay accounting belongs to the [accepted E07 report](../report/pilots/E07_seed42/E07_seed42_analysis.md) |
| [kaggle_e07_seed42_recovery_executed.ipynb](kaggle_e07_seed42_recovery_executed.ipynb) | Actual saved recovery capture; [capture method](E07_notebook_provenance.md) |
| [kaggle_e12_seed42_research_executed.ipynb](kaggle_e12_seed42_research_executed.ipynb) | Actual saved research capture; historical stdout is retained |
| [kaggle_e12_seed42_recovery_executed.ipynb](kaggle_e12_seed42_recovery_executed.ipynb) | Actual zero-update recovery capture; [capture method/SHA256](E12_notebook_provenance.md) |

Some notebook containers were reconstructed from rendered code/output blocks, not original downloaded ipynb bytes. Their metadata/provenance states that boundary; no captured code/output was edited. A pending phrase in historic stdout is not the current experiment status: see the [accepted reports](../docs/README.md) for later closure receipts.

Unique failed attempts and completed snapshots remain at their existing paths so exact identities and pinned builders stay valid. Reviewer navigation is organized here rather than moving evidence into a generic legacy folder. No new Kaggle/Colab job, final demo notebook or model code was created by cleanup.
