# Google Colab submission demo â€” genuinely executed

[Open the clean pinned notebook in Colab](https://colab.research.google.com/github/rayyan-merchant/Nano-Language-Model/blob/43b93e14326efae76a2b106dfc493d150d73f3dd/notebooks/NanoLM_Submission_Demo.ipynb).
The corrected notebook rendered in the actual browser. Its public GitHub notebook
bytes were separately retrieved anonymously from a fresh Colab environment.
The viewer used the owner's Google account; independent access is established by
anonymous public source retrieval, not by claiming a second Google account.

[Clean source](../../notebooks/NanoLM_Submission_Demo.ipynb) and
[genuine executed copy](verification/colab-final/NanoLM_Submission_Demo_executed.ipynb).
The executed copy retains actual nbclient cell outputs, timestamps and one embedded
trained attention PNG, with no code/Markdown changes from the clean source.
[Open the actual executed copy in Colab](https://colab.research.google.com/github/rayyan-merchant/Nano-Language-Model/blob/2303f56dd06ba7b857b339b3512ea185fe147163/docs/submission/verification/colab-final/NanoLM_Submission_Demo_executed.ipynb). It rendered the actual120.689-second summary and genuine outputs in the browser; anonymous raw retrieval matched the independently exported notebook SHA256 exactly. See [access receipt](verification/notebook-access.json).

## Run it

1. Open the clean link, choose a fresh CPU runtime with internet, then Runtime â†’ Run all.
2. Use your own ordinary Google sign-in to allocate free CPU compute. No HF token,
   Drive mount, D-drive path or developer setup is required.
3. Let the notebook inspect native versions, create isolated Python3.11.14 and
   install pinned CPU dependencies. Native Colab PyTorch is not replaced.
4. All23 focused tests must pass before the anonymous model downloads. Size,
   SHA256 and original identities must pass before real selected-model loading.
5. Inspect actual architecture/token IDs, trained attention, causality, all3 model
   generations, historical BPB comparison and in-process API response.
6. Download outputs/submission-colab receipts before ending the runtime. The
   immutable public inference packages can be downloaded again. Disconnect/release
   the runtime after use.

Actual successful CPU execution:13/13 code cells, no errors,106.352seconds in the
first corrected fresh runtime and120.689seconds in the final independent fresh
Jupyter execution. Allow longer for network/provider conditions; these are measured
instances, not a guaranteed live-demo time. Native Python3.13.16/torch2.11.0+cpu;
isolated inference Python3.11.14/torch2.5.1+cpu/numpy1.26.4/tokenizers0.21.0.
All30 generation cases and12/12 available canonical raw-ID matches passed twice.
[Summary](verification/colab-final/receipts/summary.json),
[environment/freeze](verification/colab-final/receipts/environment/receipt.json),
[attention](verification/colab-final/receipts/attention/trained_attention.png),
[full first-runtime suite](verification/colab-v3/full-tests.txt).

Failures/corrections and binary-MIME capture boundaries are documented in the
[execution report](reproducibility_report.md). The first two unsuccessful attempts
are not concealed. The final executed copy came directly from standard nbclient,
not reconstructed cell text or manually inserted images. Source43b93e1 CI passed
all three jobs; PR24 was subsequently merged. No GPU result is claimed.
At this reproducibility capture, the handout audit, article and recording were outside scope. The audit is now complete and the [published Medium article](https://medium.com/@negation/we-built-and-deployed-a-gpt-from-scratch-the-scores-improved-the-writing-didnt-always-follow-feaf0b8943ca) is public; the owner reports recording/presentation delivered, with evidence links pending. See [current readiness](assignment_readiness.md). This historical execution does not certify course submission.
