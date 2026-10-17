# Nano-Language-Model

A small decoder-only GPT built from scratch in PyTorch for a controlled Tiny
Shakespeare study. Attention explicitly implements Q/K/V projections, head
splitting, scaling, causal masking, softmax and value aggregation. No pretrained
language-model weights or black-box Transformer architecture are used.

**Try it:** [NanoLM Studio](https://nanolm-studio-rayyan.vercel.app) ·
[Google Colab demo](https://colab.research.google.com/github/rayyan-merchant/Nano-Language-Model/blob/main/notebooks/NanoLM_Submission_Demo.ipynb) ·
[Public pinned models](docs/submission/model_access.md)

## Results and limitations

The completed study contains **one E01 baseline plus thirteen E02–E14 controlled
variants**, each at seed 42 with 5,000 authoritative updates, and two separately
approved combined models. These are single-seed observations, not replicated
statistical evidence. The planned three-seed expansion was discontinued.

| Selected inference model | Parameters | Completed training updates | Selected update | Validation BPB |
|---|---:|---:|---:|---:|
| E02 Compact | 2,173,056 | 5,000 | 5,000 | 2.02659521 |
| B01 Balanced | 3,062,784 | 4,500, frozen early stop | 3,000 | 1.98493095 |
| F01 Quality | 6,046,560 | 10,000 | 1,750 | 1.98376817 |

[The research index](report/README.md) links every experiment, canonical tables,
figures, generation grids and checkpoint recovery instructions. Lower BPB does
not imply better writing: genuine samples include repetition, malformed words
and incomplete dialogue. F01 deteriorated after its early validation minimum.

## Install and verify

Use Python 3.10 or 3.11 in an isolated environment. For **CPU** use:

```text
python -m venv .venv
```

Activate it (`.venv\Scripts\Activate.ps1` on Windows,
`source .venv/bin/activate` on Linux), then:

```text
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r requirements-studio.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m pip check
python -m scripts.maintenance.verify_repository --output outputs/public-cleanup/verification.json
```

On Windows with a restricted system Temp directory, give pytest a new local
`--basetemp outputs/pytest-local` path. Historical-object verification requires
full Git history (`git fetch --unshallow` for a shallow clone).

Qualified GPU research used Python 3.11.14, PyTorch 2.5.1+cu124, CUDA 12.4 and a
Tesla T4. Do not install the CPU wheel into that environment. See
[platforms](docs/platforms.md) and [commands](docs/commands.md).

## Run the selected models

Download the exact public packages and run genuine CPU inference:

```text
python -m tools.submission_models --output model_artifacts
python -m tools.submission_demo --artifacts model_artifacts --section generation --output outputs/my-generation
python -m tools.submission_demo --artifacts model_artifacts --section api --output outputs/my-api
```

Existing destinations are not overwritten. For the server and frontend use the
[API guide](api/README.md) and [frontend guide](frontend/README.md).
[Submission instructions](docs/submission/README.md) include package hashes,
fresh-process checks and the executed Colab demonstration.

## Repository map

| What to inspect | Location |
|---|---|
| Explicit architecture and scientific engine | [src/nano_lm](src/nano_lm) |
| Sixteen frozen experiment definitions | [registry](experiments/registry.csv), [configs](experiments/configs) |
| Training, evaluation, resume and storage | [commands](docs/commands.md), [protocol](docs/research_protocol.md), [BPB](docs/bpb.md), [storage](docs/storage.md) |
| All research findings and numerical data | [research index](report/README.md) |
| Correctness and regression tests | [tests](tests) |
| Reusable analysis/recovery tools | [scripts](scripts/README.md), [tools](tools) |
| Ten assignment figures and handwritten correction | [evidence index](docs/submission/evidence/README.md) |
| Assignment acceptance and remaining deliveries | [readiness](docs/submission/assignment_readiness.md) |
| Historical development material and exact retrieval | [preservation contract](docs/maintenance/README.md) |

Datasets, large research checkpoints, journals and local article renders stay
outside Git. Canonical measurements and essential accepted evidence stay in the
active tree; development history is available at pinned commits. Cloning still
retrieves Git history: this cleanup reduces the active checkout, not historical
object storage.

The Medium article is under editorial preparation and is not approved for GitHub
publication. Recording, presentation, author review and instructor acceptance of
HF hosting remain separate delivery items. No new research training or reserved
test evaluation is authorized. See [AGENTS.md](AGENTS.md).
