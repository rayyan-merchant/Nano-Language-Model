# Shared-code platform entry points

kaggle_research.ipynb and colab_research.ipynb invoke the installed package only;
no model/training copy exists in notebooks. They check the locked Python profile,
prepare real data and run short CUDA verification. They never launch the main
study. See docs/platforms.md for newer Colab profiles and docs/storage.md for
persistent export/restore requirements. bootstrap.ipynb remains a synthetic
CPU foundation demonstration, not a trained-model Colab demo.
