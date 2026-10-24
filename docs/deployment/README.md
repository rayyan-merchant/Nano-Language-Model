# NanoLM Studio: operation and deployment records

Current [Git deployment profiles](vercel_git.md) and [historical hosted verification](https://github.com/rayyan-merchant/Nano-Language-Model/blob/5f3bfd8b3a95c346be9f70aa0404e1cdf2fbd430/docs/deployment/hosting.md) document the supported Git build and original rollout respectively.

Current update2026-10-15: [Studio is live](https://nanolm-studio-rayyan.vercel.app), verified by actual HTTP200 website/readiness checks. The owner authorized public HF inference packages during [submission reproducibility](../submission/README.md). Historical deployment-preparation/authorization statements below retain their original scope; public rollout is documented in [PR23](https://github.com/rayyan-merchant/Nano-Language-Model/pull/23). This task makes no additional deployment.
The scientific package and accepted evidence remain frozen. Studio is a wrapper
around canonical GPT and generation, not an alternative model implementation.
Current verification and blockers are recorded in [verification.md](https://github.com/rayyan-merchant/Nano-Language-Model/blob/5f3bfd8b3a95c346be9f70aa0404e1cdf2fbd430/docs/deployment/verification.md).

## Install and run locally

Use Python 3.10/3.11 and Node 24. Install the CPU torch wheel before the remaining
dependencies; never replace a qualified CUDA installation with a CPU build.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-studio.txt
.venv\Scripts\python -m pip install -e . --no-deps
cd frontend
npm ci
```

From the repository root, start the backend:

```powershell
$env:NANOLM_ARTIFACTS = 'ABSOLUTE_PRIVATE_PACKAGE_ROOT'
.venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

In another terminal, from `frontend`:

```powershell
$env:NEXT_PUBLIC_API_ORIGIN = 'http://127.0.0.1:8000'
$env:NEXT_TELEMETRY_DISABLED = '1'
npm run dev
```

Open `http://127.0.0.1:3000`. For production-mode local review use `npm run build`
then `npm start`. Both terminals together run the application; no cloud service,
external LLM or simulated generation is involved. Frontend configuration is baked
at build time. All model weights remain outside Git under the configured private
artifact root. Changing an environment variable does not grant public access.

## Exact selected weights and packaging

| ID | Selected step | Parameters | Accepted conditional BPB |
|---|---:|---:|---:|
| E02 | 5,000 | 2,173,056 | 2.02659521 |
| B01 | 3,000 | 3,062,784 | 1.98493095 |
| F01 | 1,750 | 6,046,560 | 1.98376817 |

Never substitute B01 4,500 or F01 10,000. Retrieve trusted originals using the
[E02 report](../../report/pilots/E02_seed42/E02_seed42_analysis.md),
[B01 retrieval guide](../../report/balanced/artifact_retrieval.md) and
[F01 retrieval guide](../../report/final/artifact_retrieval.md). Preserve existing
archives and check their manifests; do not change sharing or republish weights.

For each model, use a fresh, nonexistent model directory:

```text
python -m tools.package_models --model e02 --checkpoint TRUSTED_E02_STEP5000 --output PRIVATE_ROOT/e02
python -m tools.package_models --model b01 --checkpoint TRUSTED_B01_STEP3000 --output PRIVATE_ROOT/b01
python -m tools.package_models --model f01 --checkpoint TRUSTED_F01_STEP1750 --output PRIVATE_ROOT/f01
python -m tools.verify_studio --artifacts PRIVATE_ROOT --output NEW_INTEGRATION_RECEIPT.json
python -m tools.benchmark_studio --artifacts PRIVATE_ROOT --output NEW_BENCHMARK_RECEIPT.json
```

The converter checks the original SHA256 against committed research manifests
**before** trusted pickle loading, then selected step, configuration, tokenizer,
clean training commit and parameters. It saves float32 `model.safetensors` and
`metadata.json`, checks every tensor/dtype, exact logits and greedy/seeded token IDs
on fixed short, Unicode and cropped-context prompts. No optimizer, scheduler,
history or RNG payload is served. It refuses existing output destinations.
Metadata is the per-model provenance manifest; canonical tokenizer bytes are reused
from `assets/tokenizers`. No extra tokenizer fitting or model inference code is added.

The API loads only verified safetensors from operator-configured paths. It does
not accept filesystem paths, uploads, pickle checkpoints or external-model keys.
Normal serving cannot load the explicitly injected unit-test fixture registry.

## API and resource controls

| Endpoint | Behavior |
|---|---|
| GET /health | Liveness only; no model loads |
| GET /ready | Verifies/loads required packages; 200 only when all three are ready, otherwise 503 |
| GET /v1/models | Typed metadata and ready/not_loaded/unavailable status |
| POST /v1/generate | One genuine canonical generation with selected model identity and measured latency |
| POST /v1/compare | Sequential same-settings/same-seed inference; keeps each unavailable/failed model slot |

Default: B01, sampling, temperature 0.7, seed 42, 128 new tokens. Requests permit
1â€“256 new tokens, temperature 0.1â€“2.0, at most 4,000 prompt characters and 32 KiB
JSON bodies. Blank, surrogate, wrong-type and extra-field requests fail validation.
Responses report actual generated tokens, continuation characters, context cropping,
effective seed, latency and replacement characters. UTF-8 is decoded over complete
sequences; generated text is never cleaned up.

There is one worker and no waiting inference queue. Contention returns 429.
The bounded client-address rate limiter defaults to 20 POSTs/minute and is per
process. Run **one** Uvicorn worker; adding workers changes both memory and limits.
Liveness requests do not allocate models. Lazy loading is locked and cached.
Timeouts return 504; browser abort cancels receipt/display, while the canonical
bounded generation may finish in its occupied worker. No promise of hard interruption
is made. Streaming is omitted: the frozen generator has no tested incremental hook.
There is no fake typing, alternative sampler, KV cache, AMP or precision conversion.

Errors are structured and sanitized. Full prompts and signed artifact URLs are not
logged. CORS is explicit, not authentication. A future public edge must enforce
connection/body limits and abuse controls. Do not trust arbitrary forwarding headers;
configure a known proxy deliberately. Use TLS at the edge and a restrictive origin.

## Environment

| Variable | Default / purpose |
|---|---|
| NANOLM_ARTIFACTS | Repository model_artifacts, operator-controlled root |
| NANOLM_DEVICE | cpu; cuda requires an actually available compatible device |
| NANOLM_CPU_THREADS | 1 |
| NANOLM_TIMEOUT_SECONDS | 60 |
| NANOLM_REQUESTS_PER_MINUTE | 20 |
| NANOLM_CORS_ORIGINS | localhost:3000 and 127.0.0.1:3000 origins |
| NEXT_PUBLIC_API_ORIGIN | http://127.0.0.1:8000; rebuild frontend when changed |
| NEXT_TELEMETRY_DISABLED | Set 1 for local/CI builds |

Templates: [API](../../api/.env.example), [frontend](../../frontend/.env.example).
Uvicorn does not automatically load the template; set environment variables or use
the container environment. None of these values should contain private signed URLs.
Frontend public variables must never contain credentials.

## Tests and application builds

```text
python -m pytest -q
python -m pip check
python -m scripts.maintenance.verify_repository --output NEW_INTEGRITY_RECEIPT.json
cd frontend
npm run typecheck
npm test
npm run build
npm audit
npx playwright install chromium
npm run e2e -- studio.spec.ts
```

The full browser test requires all three real packages and both local servers.
`available-models.spec.ts` is explicitly partial B01/F01 evidence with a missing
E02 slot; it cannot certify three-model acceptance. This historical partial case
skips when all three are ready. Unit fixtures are distinct from the genuine
packaging, integration and browser receipts. On this Windows host a
fresh workspace `PWTEST_CACHE_DIR` avoids a corrupted global transform cache.

## Container and future hosting process

From repository root, with Docker available:

```text
docker build -f api/Dockerfile -t nanolm-studio:review .
docker compose up --build api
```

The CPU image runs non-root with one worker and a health check. It contains the
scientific package, canonical tokenizer and small accepted identity manifests;
no optimizer checkpoints, datasets, credentials or model weights are baked in.
Compose binds localhost only and mounts private packages read-only. CI builds the
image and verifies liveness plus truthful 503 readiness without weights. This is
container verification, not a public deployment or genuine three-model container test.

After separate approval: use Vercel for the frontend, and a private CPU container
or Modal-compatible service for the backend. Evaluate provider terms and actual
costs at that time; no accounts/resources have been provisioned here. Mount an
immutable, versioned private safetensors package root; verify hashes before exposing
readiness. Start with one CPU worker and measure cold starts, timeouts and memory on
the target platform. Local measurements are not hosting latency guarantees.

Allow at least the measured process memory plus headroom; 512 MiBâ€“1 GiB is a planning
range, not a tested container limit. GPU serving is optional and unverified here.
Confirm E02, then repeat all-model benchmarks before deciding final capacity.

Future rollout sequence: approve hosting/weights policy; privately stage packages;
build/pin image commit and lockfiles; set backend limits/TLS/proxy/CORS; verify all
three /ready and exact-model inference; build frontend with its HTTPS API origin;
verify desktop/mobile workflows in staging; review and approve public deployment.
Keep old image and immutable artifact versions for rollback. Never update weights
in place; switch to a previously verified version and recheck readiness.

## Dependencies and adapted interfaces

Serving additions are pinned separately in requirements-studio.txt; frontend uses
package-lock.json and self-hosted OFL font packages. Source package math and research
dependency pins are unchanged. Application wrappers use documented interfaces:
[FastAPI](https://fastapi.tiangolo.com/), [Next.js](https://nextjs.org/docs),
[safetensors](https://huggingface.co/docs/safetensors/index),
[Playwright](https://playwright.dev/docs/intro). No architecture or generated text
was borrowed from a pretrained model.
