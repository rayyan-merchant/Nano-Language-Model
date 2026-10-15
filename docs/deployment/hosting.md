# Authorized application hosting rollout

## Current route: Vercel-only, free Hobby

After the HF API rejected Docker cpu-basic creation with HTTP402/PRO required,
the user explicitly selected Vercel-only deployment. No HF subscription or paid
hardware was allocated. The earlier free-HF recommendation missed the eligibility
gate; current [HF rules](https://huggingface.co/docs/hub/spaces-overview) confirm it.
The original HF proposal below is retained as preparation history, not the active route.

The private HF repository rayyanmerchant/nanolm-studio-weights was successfully
published at revision719a7f3a4955225a9818e872b08c1503de601a81; all six files passed
independent authenticated read-back. See [receipt](evidence/private-model-publication.json)
and the [HF eligibility failure](evidence/hf-space-eligibility.json).

Vercel CLI64.2.0 authenticated as rayyan-merchant, email merchantrayyan43@gmail.com,
team rayyan-merchants-projects / team_zLMpuHlQoIrk6zvpwfb0vwso, verified Hobby.
The previously connected zaidbinnaveed account was not used. The connector remained
broken, so deployment uses the authenticated official CLI. User requested the
[official agent setup guide](https://vercel.com/get-started.md); global CLI installed,
existing Vercel guidance reused, shared MCP endpoint registered separately for OAuth.

Vercel requires Python3.12+; the deployment-only generated metadata pins3.12 and
PyTorch2.5.1+cpu, NumPy1.26.4 and tokenizers0.21.0. Research requirements and source
remain untouched. Hosted inference compatibility must pass before acceptance.

`tools.prepare_vercel_backend` copies clean Git source and the six hash-verified
inference files into an ignored PRIVATE deployment bundle. Model files are only
server-function assets, never public/ files or static routes. This avoids storing
a Hub token in the production runtime; the private HF repository remains a backup.
Generated serving metadata excludes research-only pytest/Matplotlib dependencies.
No quantization, architecture conversion, precision or sampling change occurs.

The frontend and API will be separate projects on the same Vercel account. Verify
the actual preview build, private-file404 responses, full three-model inference,
same-seed comparisons, cold/warm latency, errors and frontend HTTPS integration.
Only promote after these pass; no paid upgrade or merge is implied. Keep Hobby
usage limits explicit and do not promise unlimited/free high-volume inference.

## Observed Vercel deployment

API https://nanolm-studio-api.vercel.app was built from clean source
b26179c5ffac9248818775ab4d31abb5d4973208, deployment
dpl_mu1RDb8zrXzuJLxmwiWD92L31yzP. Vercel auto-assigned the project's first
deployment to production; this CLI behavior was observed, not an intentional
promotion of an untested preview. The website frontend was not deployed until
the API's real HTTPS qualification passed.

Build logs observed Python3.12, uv0.10.11 and Large Functions beta activation;
PyTorch2.5.1+cpu and other serving dependencies were pinned in generated deployment
metadata. Build succeeded in27s. Runtime memory/CPU are the Hobby function limits,
not the larger 2-core/8GiB build machine reported by CLI.

Actual HTTPS health200 (4.227s) and ready200 (3.872s) passed. All three exact model
identities loaded; six greedy/sampled32-token cases exactly matched the local
export decoded outputs, common-seed three-model comparison passed, and eight
direct model/metadata/environment/release URLs returned404. Cross-runtime raw
token IDs are not exposed by the HTTP API and are not claimed verified there.
No training or original-test evaluation occurred.

The first deployment attempt failed schema validation because includeFiles was
an array; Vercel expects one glob string. Corrected in b26179c with a regression
test. Hosting/serving suite:31 passed, one dependency warning,4.44s. Frontend proxy
configuration tests and component regressions:12 passed; TypeScript passed.
The frontend uses same-origin /api rewrites to the qualified backend; no Hub token
or model files are included in the browser project. Public frontend/browser acceptance passed as recorded below.

## Public website acceptance (2026-10-15)

Live website: [NanoLM Studio](https://nanolm-studio-rayyan.vercel.app).
Frontend source `49d6f4d8f9d00c2d74f9e2670a05e5fb7b9634eb`, deployment
`dpl_DzKRsGJGurHUEBcxnH6LfMwWMMf1`, built in 19 seconds and promoted only
following protected staging verification. The generic nanolm-studio.vercel.app
name belonged to another project; its HTTP409 refusal was preserved and it was
not overridden. The chosen free production domain needs no visitor login.
Generated deployment/team aliases retain their standard access protection.

Actual unauthenticated public desktop generation/model switching/comparison/research
passed (12.3s); public mobile and error handling passed (23.4s). The initial full
public suite's desktop API readiness preflight timed out before UI execution;
a subsequent independent readiness request returned200 in3.841s and the unchanged
desktop retry passed. These were separate passes, not one all-green public suite.
Protected staging had2 passes and1 historical partial-state skip. Screenshots
named hosted-*.png in evidence are genuine browser captures.

The public same-origin API independently passed six genuine generation comparisons,
a three-model comparison and eight private-file404 checks; see
[public API receipt](evidence/hosted-frontend-public.json) and
[release receipt](evidence/public-release.json). The source CI run38716456454
passed Python, frontend and both-container checks. All nonsecret build/runtime
settings were persisted to the two corresponding Vercel projects. No Hub token
is needed by production. Models remain private server assets and private HF backup.

Current release commands (run from clean matching source; private CLI login first):

```powershell
python -m tools.verify_hosted_studio --origin https://nanolm-studio-rayyan.vercel.app/api --deployment dpl_DzKRsGJGurHUEBcxnH6LfMwWMMf1 --output outputs/studio/deploy/public-check.json
vercel inspect dpl_DzKRsGJGurHUEBcxnH6LfMwWMMf1 --scope rayyan-merchants-projects
```

For future releases, preserve the verified IDs above, build a preview with
`--skip-domain`, qualify real inference/browser/private-file behavior, then promote
the exact tested frontend artifact. Roll back the frontend with `vercel promote`
using the prior verified deployment ID. The backend's current verified ID is
recorded above; inspect and promote only a known tested backend artifact when
needed. Do not rebuild or replace private weights merely to roll back the UI.

This is free Hobby CPU inference subject to platform quotas, cold starts,
transient readiness failures and Large Functions beta availability. One worker
and conservative request admission are suitable for a low-volume demonstration;
they do not establish distributed per-user abuse protection or high concurrency.
No paid plan, new training, original-test scoring or public weight release occurred.
The official remote Vercel MCP was registered and OAuth completed; this existing
conversation still uses CLI because its older connector has a missing-link error.

## Original HF proposal and preparation history

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

## Historical preparation status (superseded by public acceptance above)

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
`scripts.maintenance.verify_repository` also passed after the final evidence update: 1,103 protected files unchanged, 16 configurations and 15 notebooks verified; two existing frozen broken links, zero new link regressions.
