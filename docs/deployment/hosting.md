# Authorized HF backend and Vercel frontend rollout

The user authorized public application deployment on 2026-10-15. Only free CPU
Space and Vercel Hobby resources are in scope; paid plans, GPUs, public weights,
research execution and merging PRs remain outside scope. Parent application source
is a315e4cca03c8b6e2e38a3ccbb1fed3012aee0bd (PR22). Hosting changes use
codex/nanolm-hosting; scientific paths and bytes remain unchanged.

## Initial identity gates

- HF connector: rayyanmerchant, read-repos only; no local HF token was available.
- Vercel connector: zaidbinnaveed, which is not the authorized owner. No Vercel
  project, deployment, team setting or resource was created under that account.
- User was asked to reconnect Vercel privately and enable a local HF write login.
  Tokens must never be placed in chat, source, terminal arguments or receipts.

## Reproducible release sequence

1. Commit/test hosting code, confirm clean source, preserve the release SHA.
2. Privately authenticate using `.venv\Scripts\hf.exe auth login`; verify
   HfApi.whoami returns rayyanmerchant. Do not use a command-line token argument.
3. Publish only the six verified serving files to a private repository:

   ```text
   python -m tools.publish_hf_models --account rayyanmerchant --repo rayyanmerchant/nanolm-studio-weights --artifacts model_artifacts --receipt outputs/studio/deploy/private-model-publication.json
   ```

   The publisher verifies local hashes, checks the remote repository is private
   before any upload, pins the returned commit and independently downloads/checks
   every file. Existing public repositories are rejected. No original checkpoints,
   datasets, optimizer state or research journals are uploaded.
4. Create a fine-grained read token limited to that private weights repository.
   Save it directly in the Space's NANOLM_MODEL_READ_TOKEN secret. The local write
   token must not become the runtime credential. Never expose either to Vercel.
5. Build the public Space source bundle from clean committed Git objects:

   ```text
   python -m tools.prepare_hf_space --output outputs/studio/deploy/hf-source-RELEASE_ID
   ```

   This explicit allowlist excludes weights, datasets, environments, local caches,
   credentials and historical notebook outputs. release.json records source SHA
   and file hashes. The HF README exposes port8000; Docker uses UID1000 and a
   writable home/cache. The original scientific files are copied from Git, not edited.
6. Create rayyanmerchant/nanolm-studio-api as a public Docker Space on cpu-basic.
   Set NANOLM_MODEL_REPO and the exact 40-character NANOLM_MODEL_REVISION from the
   private publication receipt. Set NANOLM_MODEL_CACHE=/home/user/models and
   NANOLM_CORS_ORIGINS to the exact Vercel production origin. Upload only the
   generated source bundle. Do not allocate persistent paid storage or a GPU.
7. Space startup downloads at the pinned revision, checks all six file sizes and
   hashes, atomically promotes the complete package, validates canonical model
   identities, preloads all three, then serves the existing FastAPI application.
   Any missing/invalid private configuration fails closed with a sanitized error.
   Ephemeral storage is a cache; immutable private Hub objects are durable sources.
8. After verifying Vercel's actual user and Hobby team, create a frontend project
   with framework=nextjs, rootDirectory=frontend and Node24. Set only
   NEXT_PUBLIC_API_ORIGIN=https://rayyanmerchant-nanolm-studio-api.hf.space and
   NEXT_TELEMETRY_DISABLED=1. Confirm the actual Space host from its runtime receipt
   before using it. Deploy the exact committed feature source; no merge is implied.
9. Test public HTTPS health/readiness, genuine selected-checkpoint inference,
   same-seed comparison, private file access denial, CORS, busy429 behavior and
   a Space restart/reload. Test actual desktop/mobile frontend workflows and
   record public URLs, revisions, build IDs, timings and failures.

## Public demo limitations

This is anonymous, low-volume research inference with one bounded worker and no
waiting queue. It is not a high-concurrency service. The direct-address limiter
may apply globally to visitors behind the HF proxy; arbitrary forwarded headers
remain untrusted. Keep conservative global admission limits and display honest
busy/retry messages; do not advertise per-visitor quota isolation. Sustained public
traffic requires a separately configured trusted edge/distributed abuse limiter.

Free Spaces can sleep; the UI must distinguish waking/unavailable from inference
failure. Measure hosted cold starts and model latency before publishing readiness
claims. No prompt history or user database is enabled. Raw weights are outside the
public Space repository and are not served by an HTTP static-file endpoint.

Rollback keeps the previous tested Space source SHA, private weight revision and
Vercel deployment ID. Never replace weights in place. On a failed new build,
retain or restore the prior verified deployment; keep failure receipts.

## Current status

Hosting/serving tests passed 28 cases (9.42 s; one dependency deprecation warning),
including immutable revision enforcement, corrupt/partial-download rejection,
cached restart validation, wrong-account rejection and public-repository upload
refusal. `pip check` passed. Cloud execution and full container build are separate
acceptance gates. Preparation continues; public publication is
blocked by the account/write-authentication gates above. No successful hosted
deployment or private artifact publication is claimed until actual receipts exist.

After the user's first reconnection confirmation, the HF connector still showed
only read-repos scopes, no local CLI token existed, and Vercel returned a missing
connection-link error. These are authentication/access failures, not failed model
inference. No alternative account was used and no credentials were requested in chat.

References: [HF Docker runtime, ports and secrets](https://huggingface.co/docs/hub/spaces-sdks-docker),
[HF token permissions](https://huggingface.co/docs/hub/security-tokens),
[Vercel deployment configuration](https://vercel.com/docs/project-configuration).
