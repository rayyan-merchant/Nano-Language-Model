# Vercel Git deployment profiles

The two Vercel projects share this repository but have separate project roots:

| Project | Root Directory | Framework |
|---|---|---|
| nanolm-studio | frontend | Next.js |
| nanolm-studio-api | api | FastAPI |

Enable source files outside the root for the API build. The frontend uses the
existing NEXT_PUBLIC_API_ORIGIN=/api and NANOLM_BACKEND_ORIGIN HTTPS backend
origin. Its rewrite keeps browser calls on the frontend origin.

The API has a separate Python 3.12 CPU-only serving profile in api/pyproject.toml.
The canonical research pyproject and qualified CUDA dependencies are unchanged.
The build command stages unchanged source/configuration/tokenizer/research
identity files under ignored api/_runtime, then downloads the three public,
revision-pinned packages using tools.submission_models. Every package is checked
against the existing manifest before serving; no weights or tokens enter Git.
The entrypoint imports the existing API and scientific package from this stage.
A missing or corrupted package fails the build instead of serving a substitute.

October 18 investigation: PR28/29 previews were built at the repository root,
where automatic Python detection installed CUDA research dependencies. Builds
reported LAMBDA_SIZE_EXCEEDED at approximately 5.18 GB. The earlier successful
production deployments used locally staged serving profiles; the Git-connected
project roots had not been configured to reproduce them. New previews verify the
separate profiles; production is not promoted by this repair.
