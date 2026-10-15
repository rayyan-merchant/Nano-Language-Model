# NanoLM Studio frontend

Next.js/TypeScript application with a literary editorial visual system: self-hosted
Cormorant Garamond and Manrope, ivory surfaces, ink output panels, plum selection
and restrained gold rules. Home/writing studio, comparison and research pages are
fully responsive. Model output is literal, unedited text, never HTML or fabricated
live generation. Research metadata remains available when the API is offline.

From this directory: `npm ci`, `npm run dev`. Set NEXT_PUBLIC_API_ORIGIN before
building. `npm run typecheck`, `npm test`, `npm run build` and `npm audit` check the
application. Tests in tests/ use explicitly labeled API fixtures; e2e/studio.spec.ts
requires actual models and local servers. No three-model acceptance is implied by
fixture tests. See [local/deployment guide](../docs/deployment/README.md).

Fonts are local registry packages under SIL OFL; no runtime font CDN is required.
Rendering uses semantic labels, keyboard focus, skip navigation, reduced-motion
styles, request abort, copy feedback and honest unavailable/error/loading states.
Browser cancel does not forcibly interrupt a canonical server generation.
