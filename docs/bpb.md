# Conditional bits per byte

The source text is unmodified UTF-8. Evaluation selects the same original text
bytes for both tokenizers. Default is the full validation partition; smoke may
explicitly cap text bytes (the capped prefix ends on a valid UTF-8 boundary).
The common conditioning prefix is up to 256 original bytes, rounded down to a
complete code point. It is excluded from scored byte/token counts and likelihood.

Tokenize prefix and suffix independently and concatenate their token sequences.
There are no BOS/EOS/unknown special tokens. This forces an explicit common byte
boundary, preventing a BPE merge from straddling the scored-region boundary.
It changes the natural whole-text BPE segmentation at that boundary; this is a
conditional, boundary-constrained coding metric, not unconditional corpus entropy.

For every suffix target, sliding mode uses up to L preceding tokens, resets local
position indices for the truncated window and scores the final input position's
next-token distribution. Every target appears exactly once; likelihood sums are
weighted by tokens rather than averaging unequal batches. BPB is
sum(token NLL in nats) / (ln(2) * exactly scored original UTF-8 bytes).
Character byte lengths use UTF-8 lengths. ByteLevel BPE byte lengths count the
reversible byte-alphabet symbols in each token, including split multibyte code
points. Decoded single-token strings are not used to guess coverage. Full
reconstruction and total-byte conservation are checked for each partition.

The optional blocks mode is faster: nonoverlapping target blocks have one prior
token at block boundaries and growing within-block context. It is a different
conditioning policy and cannot be mixed with sliding results. The collector
rejects policy/byte-coverage differences. Canonical research uses full sliding
validation; smoke caps are excluded from scientific comparisons.

BPB makes a same-byte conditional comparison defensible but does not remove
context-length/tokenization, vocabulary/parameter-count, code-path or compute
confounds. Record effective byte exposure, parameters, runtime and GPU differences.
The tests compare uniform-distribution BPB with an analytic log2(vocabulary)
reference for Unicode text and compare contextual sliding likelihood with one
independent full causal forward pass. No test set is consulted for training,
tokenizer fitting, checkpoint selection or early stopping.
