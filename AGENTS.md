# Project rules

- Build our own decoder-only GPT using explicit Q/K/V, scaling, causal masking,
  softmax and value aggregation. No pretrained language-model weights or black-box
  Transformer architecture.
- GitHub main and docs/research_protocol.md are canonical. Fetch before work.
  After the coordinator's foundation, use independent descriptive feature
  branches and reviewed, tested PRs. Never force-push, overwrite another member's
  work, or merge without authorization.
- E01 and E01–E14 definitions are locked. Changes require documented user
  approval. No interaction experiments. Ownership remains unassigned until
  runtime measurements support allocation.
- Keep data partitions separate; fit BPE only on training text. Track data and
  tokenizer hashes, resolved configs, seeds, dependencies, code commits, hardware
  and checkpoint lineage. Do not select on test data.
- Never invent results or present synthetic smoke data as Tiny Shakespeare
  evidence. Record failures, limitations and affected runs after model bugs.
- Preserve checkpoints and logs in unique run directories. Never silently
  overwrite results. Resume must restore optimization and sampling/RNG state.
- Long GPU training, pilot runs and the 42-run study require subsequent explicit
  execution approval. CPU correctness and tiny overfit checks are permitted.
- Use readable modules, helpful types and meaningful tests. Update relevant
  docs for decisions, implementation, results, failures and limitations.
- Do not commit credentials, private data, dataset copies or large generated
  artifacts. Cite external sources and explain adapted ideas for the demo.
- Thin Colab/Kaggle notebooks import this package; do not duplicate model code.
- Rayyan owns the unified pipeline implementation in this Codex instance. The
  second member independently verifies and later runs the same pinned codebase.
  Do not split dataset, tokenizer, training, evaluation or checkpoint modules
  across agents. Experimental execution ownership remains to be assigned.

See docs/progress.md for implemented versus pending features and
docs/assignment_compliance.md for handout acceptance requirements.
