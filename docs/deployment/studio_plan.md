# NanoLM Studio design and implementation plan

Approved brief: three genuine selected models, private local packaging and a complete
literary research playground. Public deployment, training and test likelihood are
excluded. Starting reviewed main: 95b83f0ed8f8ac70f4b7f102a8f864acac824f66.

## Architecture and boundaries

Keep src/nano_lm, configuration paths, tokenizers and accepted evidence unchanged.
Use tools/package_models.py for trusted offline conversion: expected research-file
SHA256 before pickle loading, exact selected step/configuration/provenance, float32
safetensors plus JSON, strict reload, logits and raw generated-ID equivalence.
Artifacts stay ignored and local. Serving accepts no user files or model paths.

api/catalog.py derives the three entries from committed evidence. api/registry.py
verifies serving manifests and safetensors before constructing canonical GPTs;
lazy loading is locked and failed models remain unavailable. api/inference.py calls
the original generator, whole-sequence BPE decoding and per-request RNG. A single
bounded worker prevents model mode/device races and CPU oversubscription. Timed-out
or disconnected work retains its slot until completion; no unbounded executor queue.
Compare reports each model's error rather than substituting weights. Non-streaming
is deliberate: there is no safe streaming hook in the frozen generator.

FastAPI provides health, ready, models, generate and compare. Strict schemas,
limits, CORS, rate limits, sanitized errors and prompt-free logs surround inference.
CPU is default; optional CUDA has explicit failure, no automatic device fallback.

Next.js/TypeScript frontend: an ivory editorial landing page with charcoal navigation,
gold fine rules and plum details; a two-column writing studio, responsive three-model
comparison lab, research explanation/charts and limitations. No external LLM,
simulated generation or fake streaming. Static evidence remains available when API
is down, with a clear offline state. Central typed client and abort state; text is
rendered literally. Large serif headlines, readable UI sans and reduced-motion CSS.

## Sequential milestones and checks

- [ ] Packaging: catalog/hash tests, trusted converter, actual best-weight tensor,
  logits, greedy and seeded-ID equivalence; receipt for each accessible model.
- [ ] API: strict artifact tests, schema/resource/error tests, real-model integration,
  CPU cold/warm/64/128/256-token and compare/concurrency benchmark receipts.
- [ ] Frontend: responsive pages, accessible controls, typed API integration;
  component tests, production build, browser workflows and desktop/mobile captures.
- [ ] Delivery: scientific suite/integrity, container configuration/build if runtime
  exists, dependency locks, startup/retrieval/deployment docs and reviewable PR.

Missing checkpoints do not block independent development but do block three-model
acceptance. Docker absence is a container-verification blocker, not a passing build.
Fixtures are labeled test fixtures and cannot enter normal serving configuration.
Only genuine runtime receipts establish model integration and performance.
