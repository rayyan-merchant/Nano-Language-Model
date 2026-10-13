# E07 notebook provenance

The `*_executed.ipynb` files preserve the actual saved research and recovery notebook code, indentation, execution counts and stdout. Capture provenance and original Kaggle version URLs are in notebook metadata. These are historical execution records; do not execute them blindly. The clean qualification/research launchers are separate reproducible wrappers around the shared package. Neither notebook contains a second model or training implementation.

The original session expired. Recovery restored the verified step-2750 state and completed the same run to 5000. The recovery snapshot contains historic output stating Drive retrieval was pending at that time; the later report receipts verify its completion. Native wrapper Python and static preview GPU metadata are not the scientific environment: actual CUDA receipts prove isolated Python 3.11.14/PyTorch 2.5.1+cu124 on T4.

| Exact notebook | SHA256 |
|---|---|
| `kaggle_e07_seed42_research_executed.ipynb` | ecb9a9ab22efb80faac849eb40509120b4b9b43ccaeac166740e0502844de49b |
| `kaggle_e07_seed42_recovery_executed.ipynb` | 9612665d8d3a6c68a1aade38841afa331fbc83cb8eca19802213e12e8904ecd1 |

See ../report/pilots/E07_seed42/E07_seed42_analysis.md and its evidence for full provenance, archive recovery and replay accounting.
