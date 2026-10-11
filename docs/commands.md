# Reproducible commands and benchmark procedure

Use the installed package from a pinned clean checkout; ROOT is the repository,
DATA the prepared immutable directory and OUT a fresh run/output directory.

```text
python -m pytest -q
python -m nano_lm.prepare --output DATA
python -m nano_lm.prepare --output DATA --validate
python -m nano_lm.experiments --dataset DATA --audit --output OUT/configurations.json
python -m nano_lm.verification --dataset DATA --output OUT/cpu --device cpu
python -m nano_lm.verification --dataset DATA --output OUT/gpu --device cuda --precision float16
python -m nano_lm.experiments --dataset DATA --benchmark-all --output OUT/benchmark --device cuda --steps 10
```

Audit records actual parameter counts and shapes for all 14 IDs and checks intended
factor differences. The benchmark is 14 short 10-step jobs, not the 5k study. It
records measured steps/sec, token throughput, CUDA peak allocated memory and
projected 5k training-only seconds. Warmup is included, so a 50-step repeat can
improve the estimate after GPU qualification. Full-runtime projections must add
measured evaluation and checkpoint/export cost; do not call training-only estimates
end-to-end runtime. Compare both GPUs before assigning experiments.

After separate approval, a single research command uses --experiment E01..E14,
--seed 42/123/456, --steps 5000 and --tokens-per-update 4096 (default); AdamW,
3e-4 and float32 remain baseline. Choose microbatch to fit the actual device;
accumulation is computed exactly. Research validation is full sliding by default.
Optimizer choice, LR, weight decay, precision and evaluation overrides are logged
and must be kept controlled rather than silently mixed. Changing the stop target
to 10000 with --resume preserves the constant learning-rate regime and complete
checkpoint state. Exact-code/hardware resume is required; clean research checkout
and immutable hashes are enforced.

```text
python -m nano_lm.results RUN_A RUN_B --steps 5000 --output OUT/comparison
python -m nano_lm.plot_run RUN/checkpoints/step_005000.pt OUT/run_figures
```

Collection requires the full 42-run grid by default. --allow-incomplete permits
explicit partial tables with seed-completeness fields and undefined SD for one
seed; it still rejects incomplete individual runs and smoke/benchmark evidence.
Repeated copies of the same verified run are deduplicated. Competing runs for an
experiment/seed are rejected instead of choosing the most favorable result.
Outputs: per-run/aggregate JSON and CSV, BPB, runtime, throughput, memory,
parameter-efficiency and validation-curve PNGs. Sample SD uses ddof=1. A single-run
figure always displays its smoke/benchmark/research label. Token-loss plots are
within-tokenizer diagnostics; BPB is used for cross-tokenizer ranking.

The fixed generation prompts/temperatures/seeds/checkpoint policy are in
configs/generation_protocol.json. Test evaluation remains reserved until selection
is complete. Only load trusted project checkpoints for CLI inference.
