# Project rules

- Build our own decoder-only GPT using explicit Q/K/V, scaling, causal masking,
  softmax and value aggregation. No pretrained language-model weights or black-box
  Transformer architecture.
- GitHub main and docs/research_protocol.md are canonical. Fetch before work.
  After the coordinator's foundation, use independent descriptive feature
  branches and reviewed, tested PRs. Never force-push, overwrite another member's
  work, or merge without authorization.
- E01–E14, F01 and B01 definitions are frozen. The original single-seed
  ablations and separately approved combined F01/B01 runs are complete; the
  proposed three-seed grid was discontinued. No new research run, extension,
  interaction model or original-test assessment is authorized without explicit
  user approval. Run ownership and acceptance are recorded in individual reports.
- Keep src/nano_lm/, configs/, experiments/configs/ and assets/tokenizers/
  paths and bytes stable: research fingerprints and checkpoints depend on them.
  Do not format, relocate or rewrite scientific files for cosmetic cleanup.
  Preserve accepted evidence, notebook outputs, source commits and hashes.
- Keep data partitions separate; fit BPE only on training text. Track data and
  tokenizer hashes, resolved configs, seeds, dependencies, code commits, hardware
  and checkpoint lineage. Do not select on test data.
- Never invent results or present synthetic smoke data as Tiny Shakespeare
  evidence. Record failures, limitations and affected runs after model bugs.
- Preserve checkpoints and logs in unique run directories. Never silently
  overwrite results. Resume must restore optimization and sampling/RNG state.
- New substantive CPU/GPU training requires separate explicit execution approval.
  Ordinary CPU correctness tests and their bounded trusted fixtures are permitted.
  Model packaging, API/frontend/deployment and final submission are later tasks;
  do not advertise them as implemented merely because research is accepted.
- Use readable modules, helpful types and meaningful tests. Update relevant
  docs for decisions, implementation, results, failures and limitations.
- Do not commit credentials, private data, dataset copies or large generated
  artifacts. Cite external sources and explain adapted ideas for the demo.
- Thin Colab/Kaggle notebooks import this package; do not duplicate model code.
- Rayyan owns the unified pipeline implementation in this Codex instance. The
  second member independently verifies and later runs the same pinned codebase.
  Do not split dataset, tokenizer, training, evaluation or checkpoint modules
  across agents. Completed execution ownership is in the accepted research records.

Start with README.md and report/README.md for current status and research navigation.
Historical chronological records are retrievable through docs/maintenance/README.md.
See docs/assignment_compliance.md for completed research versus remaining
handout/submission requirements and docs/maintenance for cleanup safeguards.
