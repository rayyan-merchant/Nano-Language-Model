# Shared-code platform entry points

kaggle_research.ipynb and colab_research.ipynb invoke the installed package only;
no model/training copy exists in notebooks. They check the locked Python profile,
prepare real data and run short CUDA verification. They never launch the main
study. See docs/platforms.md for newer Colab profiles and docs/storage.md for
persistent export/restore requirements. bootstrap.ipynb remains a synthetic
CPU foundation demonstration, not a trained-model Colab demo.

## E05 seed42 notebooks

- [Executed E05 ReLU notebook](E05_seed42_ReLU_Colab_pilot.ipynb): exact snapshot
  of the Colab notebook opened in Codex's browser, with all ten cells, execution
  counts, genuine outputs, Drive report publication and final verification summary.
  This is the historical completed-run record. Its last cell records5000updates,
  20full validations,100generation labels and independent integrity/inference pass.
  [Snapshot provenance and SHA256](E05_seed42_ReLU_Colab_pilot.provenance.json).
- [Reproducible E05 launcher](colab_e05_seed42_pilot.ipynb): prepared source-only
  version with outputs cleared, a fresh Drive quota placeholder and research
  execution disabled by default. Use this for a separately authorized future run.

[Original executed Colab notebook](https://colab.research.google.com/drive/1jXE5OshAxWDrT4i0F8bw7OuFM59HXUjz).
The two files serve different purposes; the launcher does not contain the final
publication/status cells or the executed output history.

## E07 seed42 notebooks

- [Exact attempted E07 notebook](E07_seed42_EightHead_Colab_pilot.ipynb): byte-for-byte
  Drive snapshot with one provisioning cell. Colab refused T4 allocation due to
  usage limits before code execution; it contains no training outputs.
  [Provenance](E07_seed42_EightHead_Colab_pilot.provenance.json).
- [Clean E07 launcher](colab_e07_seed42_pilot.ipynb): reproducible pinned setup,
  preflight/research/recovery phases; fresh quota required and research disabled
  by default. The scoped seed42/5000 request is authorized subject to all gates.

[Original attempted E07 Colab notebook](https://colab.research.google.com/drive/1JqPgdwg__E-jLj32-ibFPM_pc6AwK1S3).
These files do not demonstrate a completed E07 run; see its partial report and
GPU allocation failure screenshot.
