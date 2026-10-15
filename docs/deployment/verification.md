# NanoLM Studio verification and acceptance

Status: **local three-model acceptance passed** after the user supplied the exact
E02 checkpoint. Public deployment remains unauthorized. This document records actual evidence;
pending checks are not claimed passed. Source baseline: reviewed main
95b83f0ed8f8ac70f4b7f102a8f864acac824f66.

## E02 blocker closure and complete local acceptance

The user supplied D:\Nano-Language-Model-artifacts\E02_seed42_checkpoint_archive_and_receipt.
Its step_005000.pt SHA256 exactly matched the accepted original:
80f2d80202f1af124fa18697258208ffc7bde9e3fa0d52009a8ceba591ab7606.
The original directory was read only. The existing converter verified step 5000,
2,173,056 parameters, clean training commit, configuration and canonical BPE hash.
Exact tensor/dtype, logits and raw greedy/seeded token-ID equivalence all passed.
No additional training or original-test assessment occurred.

The private E02 safetensors package is 8,698,368 bytes, SHA256
5258e0db6839a2ee95b24083fc4bda43299c73d79310de5e8291d6bc3e4f7150.
See [package manifest/equivalence](evidence/e02-package.json),
[six genuine API cases plus comparison](evidence/integration-three-models.json),
and [all-model CPU measurements](evidence/benchmark-three-models.json).
All six greedy/sampled API cases passed canonical decoding equivalence and /ready=200.

Actual complete browser acceptance passed: E02, B01 and F01 generation, all-three
comparison, research navigation, no page errors (15.7 s), and mobile/error workflow
(4.8 s); command total 23.9 s. Screenshots below now show all three real outputs.
The historical partial-availability test is conditional when all three are ready;
it does not replace the strict full acceptance test, which ran and passed.
The subsequent complete browser-suite command passed 2 tests and skipped only that
historical partial-state case in 27.9 s. A refreshed
[integrity audit](evidence/integrity-three-models.json) again verified all 1,103
protected files, 16 configurations and 15 notebooks, with only the two existing
frozen broken links. No runtime/scientific implementation changed during closure.

Same local CPU/Python/torch profile as below, three warm repetitions per length:

| Model | 64-token median | 128-token median | 256-token median | Cold load |
|---|---:|---:|---:|---:|
| E02 | 0.447 s | 1.628 s | 4.662 s | 0.114 s |
| B01 | 2.246 s | 3.479 s | 7.822 s | 0.528 s |
| F01 | 1.774 s | 12.371 s | 21.206 s | 0.877 s |

Complete three-model 128-token comparison: 11.167 s. Concurrent requests returned
200/429. Peak process working set: 286,830,592 bytes (273.54 MiB). System timing
variation is visible against the prior two-model receipt; these repetitions do not
establish stable hardware-independent throughput or model superiority. No overlapping
build, test or browser inference was launched during this benchmark. Cloud resource
limits, optional CUDA serving and public deployment remain unverified/unapproved.

## Initial two-model verification history (superseded availability)

B01 step 3000 and F01 step 1750 were discovered in the independently recovered D:
artifact locations documented by their original retrieval guides. Their original
research-file hashes matched before trusted loading. No source checkpoint changed.

| Model | Inference bytes | safetensors SHA256 |
|---|---:|---|
| B01 | 12,260,216 | 0ea6ff2972740d2775f5d2da331d41395e2f75b36076436ec0b2a43ebcf18e3f |
| F01 | 24,198,360 | b1e3d1c1f6c5386599b14e5311d59cc27d72743c6dfc5ea2dc9080fef8b09238 |
| E02 | Not packaged | Required trusted original unavailable locally |

Both actual exports passed exact tensor/dtype, logits and raw greedy/seeded token-ID
comparisons for fixed short, Unicode and long/cropped prompts. The canonical BPE
hash is 2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55.
Their selected counts remain 3,062,784 and 6,046,560. Packages retain no optimizer,
RNG, scheduler or training journal. Generated output was not rewritten.

The E02 selected original must match
80f2d80202f1af124fa18697258208ffc7bde9e3fa0d52009a8ceba591ab7606.
No matching research checkpoint/archive was found in the documented local artifact
roots. Its published Drive link requires sign-in in the available browser. A local
path was requested; credentials were not requested, used or recorded. Random
weights and alternate steps were never substituted.

## Actual checks so far

- Final full Python suite: 140 passed, 1 CUDA-only skipped, 15 dependency warnings,
  362.20 s; pip check passed. Earlier pre-size-cap suite passed 139/1 in 197.41 s.
- New serving suite including artifact-size rejection: 19 passed, 1 test-client warning,
  5.12 s. Fast fixtures are explicitly injected and are not research-weight evidence.
- TypeScript, seven frontend component tests, production Next.js build and npm audit
  passed; audit reported zero vulnerabilities after a targeted Vitest update.
- Genuine API integration passed greedy/sampled B01/F01 full-sequence decoding
  comparisons; E02 remained unavailable and /ready returned 503. Complete actual
  API responses are in the integration receipt.
- Offline integrity passed all 1,103 protected paths, all 16 configurations,
  both tokenizers and all 15 notebooks; the two pre-existing frozen links remain.
- Docker is unavailable locally. Linux CI actually built the image, ran /health
  successfully and confirmed /ready=503 without private weights. No image, weights
  or application was published.
- Genuine two-model desktop browser workflow passed (20.0 s test / 29.5 s command),
  including switching, actual inference, exact clipboard content allowing only native
  Windows LF/CRLF conversion, comparison and research navigation, with no page errors.
- Mobile browser workflow passed (6.6 s test / 21.6 s command), including genuine B01,
  390 px layout, 320 px overflow check and an intentionally aborted request/error state.
- Complete three-model browser acceptance was actually executed and failed at its
  required readiness gate: expected 200, received 503. This is the missing E02 blocker.
- [CI run 38677912412](https://github.com/rayyan-merchant/Nano-Language-Model/actions/runs/38677912412)
  passed all three jobs (Python/integrity, frontend and container) for source
  8198882b3cf4a03cb3bb1da2105dee7ae631ecd2. Later changes only document these receipts
  and extend browser keyboard coverage; their head CI is inspected separately in PR22.
  CI logs record 140 passed, 1 skipped, 15 warnings in 37.96 s, and the same protected
  file/configuration/notebook counts and two pre-existing frozen broken links.
- A final genuine desktop repetition activated Generate with keyboard Enter for
  both available models and passed (14.0 s test / 18.0 s command).

## Genuine evidence and measured CPU performance

Receipts: [B01 package](evidence/b01-package.json), [F01 package](evidence/f01-package.json),
[integration](evidence/integration.json), [benchmark](evidence/benchmark.json),
[protected-file audit](evidence/integrity.json). These contain real unedited generations,
not unit fixtures. Per-model metadata is the inference manifest; no root index is claimed.

Windows Intel Core i3-10110U, 2 cores/4 logical processors, one torch thread;
Python 3.10.11, PyTorch 2.5.1+cpu. Three warm repetitions after an 8-token warmup:

| Model | 64-token median | 128-token median | 256-token median | Cold model load |
|---|---:|---:|---:|---:|
| B01 | 0.778 s / 82.3 tok/s | 2.599 s / 49.2 tok/s | 11.782 s / 21.7 tok/s | 0.141 s |
| F01 | 3.616 s / 17.7 tok/s | 8.025 s / 15.9 tok/s | 11.852 s / 21.6 tok/s | 0.620 s |
| E02 | unavailable | unavailable | unavailable | unavailable |

The 128-token comparison measured 8.179 s for two genuine results plus an unavailable
E02 slot. Concurrent requests returned 200 and 429. Process peak working set was
276,144,128 bytes (263.35 MiB), not GPU memory or a cloud capacity guarantee.
Timings vary with system load; earlier noisy attempts remain in ignored local outputs.
No GPU was allocated. Local CPU B01 is usable at the default length; cloud latency,
all-three resource requirements and optional CUDA serving remain unverified.

Visually inspected actual captures: [home](evidence/desktop-home.png),
[playground](evidence/desktop-playground.png), [comparison](evidence/desktop-compare.png),
[research](evidence/desktop-research.png), [mobile](evidence/mobile-playground.png),
[mobile error](evidence/mobile-error.png). Self-hosted serif/sans typography, ivory,
charcoal, plum and gold styling; keyboard skip link, labels, focus states and reduced
motion are implemented. No comprehensive screen-reader qualification is claimed.

## Independent code review

Read-only reviews of code commit 0f9576a against reviewed main 95b83f0:

| Standards review | Specification review |
|---|---|
| 0 blocking findings; preservation, artifact safety and bounded worker behavior checked | 0 scientific changes/scope creep; 3 actionable UI/verification gaps |
| No concrete maintainability smell requiring restructuring | Actual settings now displayed; control/loading test added; comparison loading clarified |

Comparison shows aggregate loading until the non-streaming sequential response arrives.
It cannot report an individual model's intermediate completion. No simulated progress,
streaming or token animation is presented. Review is code evidence, not human PR approval.

## Operational findings

The first converter assumed a nonexistent metadata.source.commit field; research
metadata stores Git lineage under environment.git_commit. The wrapper was corrected
without changing the source checkpoint. An escaped-surrogate API test initially
failed in the HTTP client before sending; it now submits escaped JSON to exercise
server rejection. TypeScript fixture IDs were corrected to the literal model union.

Initial frontend test invocation hit sandbox filesystem restrictions. A later
Playwright global transform cache returned corrupt bytes while all source files
were intact; a fresh workspace cache resolved startup. The first browser comparison
selected settings before navigation hydration, observed the default 128-token
request and failed its 64-token assertion. Controls now wait during initial readiness
and the unchanged expected-token assertion waits for the connected state.

Final browser checks also found native Windows clipboard LF-to-CRLF conversion and
a strict locator matching both the app alert and Next.js route announcer. Assertions
now allow only OS newline conversion and target the application alert specifically.

npm audit identified vulnerable Vitest test-tool versions. Vitest is now pinned to
the exact patched 4.1.11 version identified by the advisory; no forced global upgrade
or scientific dependency change was made. Current audit is clean.

Serving is non-streaming, one bounded worker. Browser cancellation cannot forcibly
interrupt the frozen generator; the server keeps its slot until completion. Optional
CUDA inference and target-cloud cold starts are unverified. Unit fixtures and image
liveness without weights cannot establish genuine three-model acceptance.
