# NanoLM · GPT from scratch

**A small language model, sixteen measured architectures, and a three-model playground.**

We built a decoder-only Transformer from first principles, trained it on Tiny
Shakespeare, and studied how tokenization, positional encoding, attention heads,
depth, context length and model width affect prediction and generated text.
The result is an inspectable research project and a working application—not a
pretrained model wrapper.

[![Open NanoLM Studio](assets/readme/studio.svg)](https://nanolm-studio-rayyan.vercel.app)
[![Read the Medium report](assets/readme/medium.svg)](https://medium.com/@negation/we-built-and-deployed-a-gpt-from-scratch-the-scores-improved-the-writing-didnt-always-follow-feaf0b8943ca)
[![Get the model artifacts](assets/readme/models.svg)](https://huggingface.co/rayyanmerchant/nanolm-studio-weights/tree/719a7f3a4955225a9818e872b08c1503de601a81)
[![Run the reproducibility notebook](assets/readme/colab.svg)](https://colab.research.google.com/github/rayyan-merchant/Nano-Language-Model/blob/main/notebooks/NanoLM_Submission_Demo.ipynb)

**Jump to:** [The project](#the-project) · [Try the models](#try-the-models) ·
[All 16 experiments](#all-16-experiments) · [Repository guide](#repository-guide) ·
[Assignment evidence](#assignment-evidence) · [Team](#team) · [Reproduce the work](#reproduce-the-work)

## Team

| Name | Roll number |
|---|---|
| Rayyan Merchant | 23K-0073 |
| Syeda Rija Ali | 23K-0057 |
| Riya Bhart | 23K-0063 |

## The project

This repository accompanies the **AI Engineering — Transformer & GPT From Scratch**
assignment. The implementation explicitly computes Q/K/V projections, scaled
attention scores, causal masking, softmax and value aggregation. It includes
multi-head attention, positional encodings, pre-LayerNorm residual blocks,
feed-forward networks, shifted next-token training and autoregressive generation.
No pretrained language-model weights or black-box Transformer layers are used.

```text
Text → character or BPE tokens → embeddings + positional information
     → causal Transformer blocks → vocabulary logits → next token
```

The corpus is [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt):
1,115,394 original bytes, partitioned contiguously into 90% training, 5% validation
and 5% test. Tokenizers fit training text only. See [dataset provenance](data/README.md).

**The central finding:** better validation scores did not consistently produce
better writing. The models learn recognizable words and dialogue structure, but
repetition, malformed text and weak long-range coherence remain visible in the
[unedited generations](docs/submission/evidence/figure_08_generations.md).

## Try the models

Open [NanoLM Studio](https://nanolm-studio-rayyan.vercel.app) to generate text or
compare the three selected models with the same prompt:

| Model | Selected checkpoint | What it represents |
|---|---:|---|
| **Compact · E02** | 5,000 | Smallest deployed model; baseline architecture with byte BPE tokenization |
| **Balanced · B01** | 3,000 | Six blocks and RoPE; nearly F01's best validation BPB with about half the parameters |
| **Quality · F01** | 1,750 | Lowest observed validation BPB; larger combined architecture, selected before late deterioration |

These are model labels, not a guaranteed ranking of writing quality. All three
use their own trained weights. The [public Hugging Face packages](https://huggingface.co/rayyanmerchant/nanolm-studio-weights/tree/719a7f3a4955225a9818e872b08c1503de601a81)
contain inference weights and provenance; [model access instructions](docs/submission/model_access.md)
provide hashes and original checkpoint identities.

Prefer a reproducible demonstration? [Open the Colab notebook](https://colab.research.google.com/github/rayyan-merchant/Nano-Language-Model/blob/main/notebooks/NanoLM_Submission_Demo.ipynb).
It downloads the pinned models, checks hashes, visualizes real trained attention
and runs genuine inference without retraining. [Executed notebook and receipts](docs/submission/verification/colab-final/NanoLM_Submission_Demo_executed.ipynb)
are retained for inspection.

## All 16 experiments

**Click an experiment ID for its full report; click its description for the exact
configuration.** E01 is the baseline; E02–E14 change one intended factor relative
to E01. F01 and B01 combine factors and are separate model-design experiments.

**BPB = conditional validation bits per original byte; lower is better.**
“Best update” identifies the validation-selected checkpoint. “Final BPB” is the
score at the end of training, which can be worse than the selected checkpoint.
All results below are genuine **seed-42 observations**, rounded to five decimals.

| Experiment / report | Configuration or intended change | Parameters | Completed updates | Best update | Best BPB ↓ | Final BPB ↓ |
|---|---|---:|---:|---:|---:|---:|
| [E01](report/pilots/E01_seed42/E01_seed42_analysis.md) | [Baseline: 4 blocks, width 192, 6 heads, context 128](experiments/configs/E01.json) | 1,804,800 | 5,000 | 4,750 | 2.21406 | 2.21600 |
| [E02](report/pilots/E02_seed42/E02_seed42_analysis.md) | [Byte BPE tokenizer · Compact model](experiments/configs/E02.json) | 2,173,056 | 5,000 | 5,000 | 2.02660 | 2.02660 |
| [E03](report/pilots/E03_seed42/E03_seed42_analysis.md) | [RoPE positional encoding](experiments/configs/E03.json) | 1,804,800 | 5,000 | 5,000 | 2.07327 | 2.07327 |
| [E04](report/pilots/E04_seed42/E04_seed42_analysis.md) | [Learned positional embeddings](experiments/configs/E04.json) | 1,829,376 | 5,000 | 5,000 | 2.23779 | 2.23779 |
| [E05](report/pilots/E05_seed42/E05_seed42_analysis.md) | [ReLU instead of GELU](experiments/configs/E05.json) | 1,804,800 | 5,000 | 5,000 | 2.24267 | 2.24267 |
| [E06](report/pilots/E06_seed42/E06_seed42_analysis.md) | [4 attention heads](experiments/configs/E06.json) | 1,804,800 | 5,000 | 4,750 | 2.20538 | 2.20618 |
| [E07](report/pilots/E07_seed42/E07_seed42_analysis.md) | [8 attention heads](experiments/configs/E07.json) | 1,804,800 | 5,000 | 5,000 | 2.21052 | 2.21052 |
| [E08](report/pilots/E08_seed42/E08_seed42_analysis.md) | [6 Transformer blocks](experiments/configs/E08.json) | 2,694,528 | 5,000 | 4,750 | 2.14549 | 2.14758 |
| [E09](report/pilots/E09_seed42/E09_seed42_analysis.md) | [8 Transformer blocks](experiments/configs/E09.json) | 3,584,256 | 5,000 | 4,750 | 2.10015 | 2.11681 |
| [E10](report/pilots/E10_seed42/E10_seed42_analysis.md) | [2× MLP expansion](experiments/configs/E10.json) | 1,213,440 | 5,000 | 5,000 | 2.26712 | 2.26712 |
| [E11](report/pilots/E11_seed42/E11_seed42_analysis.md) | [64-token context](experiments/configs/E11.json) | 1,804,800 | 5,000 | 5,000 | 2.15396 | 2.15396 |
| [E12](report/pilots/E12_seed42/E12_seed42_analysis.md) | [256-token context](experiments/configs/E12.json) | 1,804,800 | 5,000 | 5,000 | 2.25628 | 2.25628 |
| [E13](report/pilots/E13_seed42/E13_seed42_analysis.md) | [Model width 144](experiments/configs/E13.json) | 1,021,824 | 5,000 | 5,000 | 2.32496 | 2.32496 |
| [E14](report/pilots/E14_seed42/E14_seed42_analysis.md) | [Model width 240](experiments/configs/E14.json) | 2,808,960 | 5,000 | 4,750 | 2.14189 | 2.14506 |
| [F01](report/final/F01_10000_analysis.md) | [Combined: BPE + RoPE + 8 blocks + width 240](experiments/configs/F01.json) | 6,046,560 | 10,000 | 1,750 | 1.98377 | 3.05346 |
| [B01](report/balanced/B01_seed42_analysis.md) | [Combined: BPE + RoPE + 6 blocks · Balanced model](experiments/configs/B01.json) | 3,062,784 | 4,500 | 3,000 | 1.98493 | 2.04446 |

**Read the comparisons:** [E01–E14 analysis](report/final/E01_E14_consolidated_analysis.md) ·
[F01 interpretation](report/final/interpretation.md) ·
[B01 model selection](report/balanced/B01_model_selection.md) ·
[Full research index and raw measurements](report/README.md)

The pilots each completed 5,000 authoritative updates and 20 full validations.
F01 completed 10,000 updates; B01's predefined stopping rule ended at 4,500.
Replay accounting, where applicable, is retained in each report. The planned
three-seed expansion was discontinued, so these observations do not establish
statistical significance. Tokenizers, capacity, context, byte exposure, compute
and stopping opportunities differ; see the [comparison policy](docs/bpb.md).
F01 received one frozen test assessment; the table uses validation results,
and the original test was not reused to select B01.

## Repository guide

| I want to… | Start here |
|---|---|
| Read the story, methodology and conclusions | [Medium report](https://medium.com/@negation/we-built-and-deployed-a-gpt-from-scratch-the-scores-improved-the-writing-didnt-always-follow-feaf0b8943ca) |
| Understand the actual Transformer | [Model](src/nano_lm/model.py), [attention](src/nano_lm/attention.py), [positions](src/nano_lm/positions.py) |
| Inspect the data and language-modeling pipeline | [Data provenance](data/README.md), [preparation](src/nano_lm/prepare.py), [training](src/nano_lm/train.py), [evaluation](src/nano_lm/evaluate.py) |
| Find an experiment, plot, generation grid or recovery record | [Research index](report/README.md) |
| Check what changed between architectures | [Experiment registry](experiments/registry.csv), [all configurations](experiments/configs) |
| Inspect correctness and numerical checks | [Tests](tests), [submission verification](docs/submission/reproducibility_report.md) |
| Retrieve selected models or full research checkpoints | [Selected model access](docs/submission/model_access.md), [F01 recovery](report/final/artifact_retrieval.md), [B01 recovery](report/balanced/artifact_retrieval.md); pilot instructions are in each report |
| Understand the web application | [FastAPI backend](api/README.md), [frontend](frontend/README.md), [deployment](docs/deployment/README.md) |
| Find original development material removed during cleanup | [Pinned historical retrieval](docs/maintenance/README.md) |

## Assignment evidence

The [ten-figure evidence index](docs/submission/evidence/README.md) links the
hand-calculated attention derivation, causal mask, real tensor shapes, sinusoidal
encoding, architecture and parameter counts, learning curves, trained attention,
genuine text samples, controlled ablations and actual API inference.

- **Mathematical understanding:** [original handwritten work](docs/handwritten-assignment.pdf)
  and [independently checked corrections](docs/submission/evidence/hand_calculation_audit.md).
- **Seven required concepts:** [conceptual coverage](docs/submission/evidence/concepts_and_objectives.md),
  including the [MLM-versus-CLM comparison](docs/submission/evidence/mlm_vs_clm.png).
- **Reproducibility:** [clean Colab entry point](notebooks/NanoLM_Submission_Demo.ipynb),
  [executed acceptance evidence](docs/submission/reproducibility_report.md) and
  [public model identities](docs/submission/models.json).
- **Requirement-by-requirement audit:** [compliance matrix](docs/submission/assignment_compliance_matrix.md)
  and [remaining submission actions](docs/submission/assignment_readiness.md).

## Reproduce the work

The Colab button above is the easiest way to inspect the trained models. Local
setup, training and inference details are collapsed below to keep this page focused
on navigation; they are included because the assignment explicitly requires them.

<details>
<summary><strong>Run locally — setup, training, inference and expected output</strong></summary>

Use Python 3.10 or 3.11. Create and activate an isolated environment:

```text
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`.
Linux/macOS: `source .venv/bin/activate`.
For CPU inference and checks:

```text
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r requirements-studio.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m tools.submission_models --output model_artifacts
python -m tools.submission_demo --artifacts model_artifacts --section generation --output outputs/my-generation
python -m tools.submission_demo --artifacts model_artifacts --section api --output outputs/my-api
```

Expected output: hash-verified E02/B01/F01 packages, recorded generation results in
`outputs/my-generation/receipt.json`, and an actual B01 HTTP-200 API result in
`outputs/my-api/receipt.json`. Use new output directories; previous evidence is
not overwritten. A recorded B01 greedy output is:

```text
ROMEO:
I'll tell thee, I'll not be a king,
And nothing but a man.

JULIET:
I know not, sir, if
```

This is genuine, unedited model text; the incomplete ending is intentional.
See [all five recorded examples and decoding settings](docs/submission/evidence/figure_08_generations.md).
On Windows with a restricted system Temp directory, use a fresh
`--basetemp outputs/pytest-local` for pytest.

**Historical training recipe:** prepare the pinned corpus, then invoke the same
shared engine. The example documents E01; exact historical reproduction requires
its report's source commit, qualified CUDA environment and persistent storage.
It is not a request to start a new research run.

```text
python -m nano_lm.prepare --output data/tiny-shakespeare
python -m nano_lm.prepare --output data/tiny-shakespeare --validate
python -m nano_lm.train --dataset data/tiny-shakespeare --experiment E01 --seed 42 --steps 5000 --device cuda --precision float32 --micro-batch-size 4 --tokens-per-update 4096 --eval-every 250 --checkpoint-every 500 --output outputs/reproduction/E01 --export-root outputs/reproduction/snapshots
```

Do not install a CPU-only PyTorch build into a CUDA research environment.
[Detailed commands](docs/commands.md), [platform setup](docs/platforms.md),
[research protocol](docs/research_protocol.md) and [durable storage](docs/storage.md)
cover the complete reproduction and recovery workflow. The local export path
above must be replaced by a verified persistent destination for a real cloud run.
No new research training or reserved-test assessment is authorized by these examples.

</details>

**Known limitations:** one seed per architecture, a narrow corpus, unequal
computation/byte exposure, and generated repetition or malformed text. A lower
validation minimum does not prove better language quality or general superiority.
Model hosting availability can change. Instructor acceptance of HF as the
checkpoint-sharing route, author review, recording and presentation remain separate
submission decisions; dated audit records describe their capture-time status.
