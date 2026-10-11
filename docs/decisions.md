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
