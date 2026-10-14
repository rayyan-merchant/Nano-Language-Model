# Current execution status and original experiment design

E01-E14 have one completed seed42 trajectory each. The original seeds42/123/456 grid was discontinued; no missing seed is fabricated. F01/B01 are separately approved combined models, not single-factor ablations. [Current reports](README.md) record actual owners, source hashes, acceptance and limitations.

The locked registry/configuration definition fields and original design below are preserved creation/planning records, not a live job-status database. Do not change immutable config metadata to relabel completed runs.

---

# Experiment design

Canonical definitions: `experiments/registry.csv` and corresponding JSON overrides.
E01 baseline; E02 BPE; E03 RoPE; E04 learned positions; E05 ReLU; E06 4 heads;
E07 8 heads; E08 6 blocks; E09 8 blocks; E10 2x MLP; E11 context 64; E12 context
256; E13 width 144; E14 width 240. Every ID uses seeds 42,123,456.

Question: how does each factor change validation modeling quality, runtime,
memory, parameter efficiency and fixed-prompt generation under controlled
training? Hypothesis (unconfirmed): extra width/depth may improve capacity at
higher compute cost; longer context may aid dependencies but changes memory;
position methods, activation and head count may alter learning; BPE changes
sequence compression. No direction of effect is asserted as a result.

For each ID record hypothesis, motivation, exact override, controls, method,
actual measurements, plots, observations, interpretation, confounds, conclusions
and references in a run-linked research record. Parameter count is measured for
all IDs; width runs deliberately vary capacity. Derived MLP sizes: E13=576,
E14=960; E10=384. RoPE rotates Q/K, not V, with even head dimensions.
Owner/status fields remain UNASSIGNED/DEFINED_NOT_RUN; do not assign a 7/7 split
before profiling actual configuration runtimes and both GPU environments.
