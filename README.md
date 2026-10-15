# Nano-Language-Model

[NanoLM Studio](https://nanolm-studio-rayyan.vercel.app) is deployed with genuine
E02/B01/F01 inference (public website and readiness rechecked2026-10-15).
The [submission workflow](docs/submission/README.md) retrieves the exact selected
public inference packages without authentication. Clean-clone and Colab acceptance
are tracked separately in the [execution report](docs/submission/reproducibility_report.md).
Deployment implementation/evidence remains in [PR23](https://github.com/rayyan-merchant/Nano-Language-Model/pull/23).

A decoder-only GPT built from scratch in PyTorch for a controlled Tiny Shakespeare study. Attention explicitly implements Q/K/V projections, head splitting, scaled scores, causal masking, softmax and value aggregation. There are no pretrained language-model weights or black-box Transformer layers.

## Completed research and remaining delivery work

**E01–E14 seed 42 are complete:** fourteen single-factor ablations, each with a 5,000-update authoritative trajectory and 20 full sliding validations. The proposed three-seed/42-run expansion was discontinued; no replicated significance is claimed. Two separately approved combined models also completed and passed independent artifact and compatible CPU/CUDA recovery:

| Model | Parameters | Actual updates | Best update | Best conditional BPB |
|---|---:|---:|---:|---:|
| E02 BPE reference | 2,173,056 | 5,000 | 5,000 | 2.02659521 |
| F01 quality-focused | 6,046,560 | 10,000 | 1,750 | 1.98376817 |
| B01 balanced | 3,062,784 | 4,500, frozen early stop | 3,000 | 1.98493095 |

All reports retain negative results, genuine generations and recovery evidence. F01 deteriorated with prolonged training; B01 nearly matches its selected likelihood at roughly half its parameters but still produces greedy loops, malformed language and invalid byte sequences. Smaller size or lower BPB does not establish better text quality. Model packaging, FastAPI, frontend and deployment are implemented. The curated submission notebook/reproducibility work is tracked separately; the final academic report, comprehensive assignment-evidence audit and recorded presentation remain pending. No further research training or test evaluation is currently authorized.

Start with the [project map and current status](docs/README.md), [fourteen-run analysis](report/final/E01_E14_consolidated_analysis.md), [F01 report](report/final/F01_10000_analysis.md) and [B01 model selection](report/balanced/B01_model_selection.md).

## Installation and verification

Research pins: Python 3.10/3.11, PyTorch 2.5.1, NumPy 1.26.4, tokenizers 0.21.0, Matplotlib 3.9.4 and pytest 8.3.4. Use an isolated environment. For local **CPU** work:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m pip check
```

On Linux use the selected environment's `python`. Qualified GPU research used isolated Python 3.11.14/PyTorch 2.5.1+cu124/CUDA 12.4/Tesla T4; **do not install the CPU wheel into a CUDA environment**. See [platforms](docs/platforms.md) and the [qualified dependency freeze](docs/qualifications/evidence/rayyan_requirements_freeze.txt).

## Scientific package and commands

The authoritative implementation is [src/nano_lm](src/nano_lm). Stable configuration paths are [configs](configs) and [experiments/configs](experiments/configs). Character vocabulary 65 and byte BPE 1024 fit only training text; canonical artifacts are in [assets/tokenizers](assets/tokenizers). Pinned original text has contiguous 90/5/5 partitions with no crossing windows; see [data provenance](data/README.md).

The following are interface/reproducibility examples, not authorization to launch another scientific run:

```text
python -m nano_lm.prepare --help
python -m nano_lm.train --help
python -m nano_lm.verification --help
python -m nano_lm.generate --help
python -m nano_lm.storage --help
```

For a fresh local development dataset, `python -m nano_lm.prepare --output DATA` followed by `--output DATA --validate` acquires and checks the pinned corpus. Research reproduction must use its exact trained commit, configuration and tokenizer; F01/B01 reuse audited vocabulary artifacts without refitting. Generation from a trusted recovered model uses:

```text
python -m nano_lm.generate --checkpoint CHECKPOINT --prompt "ROMEO:" --max-new-tokens 256 --greedy
```

Complete commands and resume/storage boundaries are in [docs/commands.md](docs/commands.md) and [scripts/README.md](scripts/README.md). Historical notebooks and failed attempts are indexed in [notebooks/README.md](notebooks/README.md); do not blindly rerun them.

Evaluation is the reusable Python function `nano_lm.evaluate.evaluate`, called by
training and verification. It has no standalone evaluation CLI.

## Results, checkpoints and reproducibility

[report/pilots](report/pilots) contains E01–E14, [report/final](report/final) F01, and [report/balanced](report/balanced) B01. Figures, machine-readable tables, complete fixed generation grids, hashes and genuine recovery receipts are preserved. Large datasets, full checkpoints, archives and training journals are **outside Git**. For retrieval use each pilot report, [F01 instructions](report/final/artifact_retrieval.md) or [B01 instructions](report/balanced/artifact_retrieval.md). Only deserialize trusted project checkpoints; optimizer/RNG payloads use PyTorch pickle.

Conditional BPB scores the same 55,514 validation bytes after a 256-byte conditioning exclusion, with each model's context and explicit tokenizer-boundary policy. Token loss/perplexity is not comparable across tokenizers. Equal target-token budgets do not imply equal compute/source exposure. One seed supplies no cross-seed variance; initialization, capacity, context, hardware and stopping opportunities are confounders. F01 received one frozen reserved-test assessment; B01 did not reuse that test for tuning or evaluation. See [BPB policy](docs/bpb.md) and [research protocol](docs/research_protocol.md).

The [cleanup verification](docs/maintenance/cleanup_verification.md) records source, configuration, tokenizer, evidence and notebook integrity plus build/link checks. [AGENTS.md](AGENTS.md) governs future changes. Submit reviewed feature PRs; never force-push, overwrite evidence or merge without authorization.
