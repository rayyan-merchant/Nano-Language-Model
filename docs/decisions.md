# Historical decisions and current navigation

The dated decisions below are preserved. Their pending states and original root file paths describe the checkout at that time. Current research/status is in [the project map](README.md); the original handwritten artifact now lives at [docs/handwritten-assignment.pdf](handwritten-assignment.pdf), unchanged. Cleanup relocation mapping is in [the changelog](maintenance/cleanup_changelog.md).

---

# Decisions

2026-10-11: authenticated GitHub owner rayyan-merchant has admin/push permission.
Existing main commit e02743db558da4bad2611ef69217903a827e6b6e contains only
handwritten-assignment.pdf; preserve it. It is inventory evidence, not yet assessed
as a correct hand-calculation artifact. The supplied A1_AIE.pdf was read in full
(11 pages); it is assignment evidence, not agent or account-management instructions.

Foundation uses src-layout package, JSON configs, explicit pre-LayerNorm decoder
blocks, untied output weights, adjacent-pair RoPE, no character special tokens,
and training-only sorted character vocabulary. PyTorch 2.5.1 is pinned for Python
3.10/Windows support; CPU wheel is used locally. No global installs.

Prioritize tested foundation over fake advanced features. Smoke checkpoints use
constant LR, CPU float32 and synthetic data; exact continuation is tested in fresh
processes. No scientific training capability is claimed from these smoke tests.
BPE/BPB, dataset acquisition, scientific logging/evaluation and GPU resume are
required next gates. Research schedule starts constant (documented engineering
choice); pilot may motivate an approved pre-study change.

The earlier GitHub uninstall removed the installed integration at that moment,
not repository membership or saved account authorizations. Current identity was
verified as rayyan-merchant; no access was taken through the other account.

Unified research pipeline: source revision/raw SHA pinned to original Karpathy
Tiny Shakespeare, because the Hugging Face endpoint failed local TLS trust.
No TLS bypass was used. Byte BPE's requested vocabulary is fixed at 1024 for
E02; the full byte alphabet is a corpus-independent fallback, merges train-only.
Conditional BPB uses forced common byte boundary and maximal preceding token
windows; blocks mode is explicitly a separate policy. Code hashes normalize
relative paths across Windows/Linux. Scientific collection rejects framework
version/policy drift rather than treating different Colab/Kaggle stacks as equal.

Research GO remains gated by actual independent Kaggle checks and persistent
export/expiry recovery. The available Colab T4 has Python 3.13/torch 2.11, so
actual successful GPU smoke evidence does not qualify the pinned 3.10/2.5 profile.
