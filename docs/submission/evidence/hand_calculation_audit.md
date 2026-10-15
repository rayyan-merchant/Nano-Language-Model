# Hand-calculation verification and corrected companion

Audit date: 2026-10-15. The original [four-page handwritten PDF](../../handwritten-assignment.pdf)
is preserved, SHA256 `317fb3479a7629d37e9b38bf5ab227ede15801abfef18af4f4e33e5244951c3e`.
All four pages were rendered and visually inspected. This companion is typed
educational evidence prepared after the research; it is not a newly handwritten
document or proof of any student's personal understanding before training.

## What the original demonstrates

| Original page | Content inspected | Outcome |
|---|---|---|
| 1 | Three tokens, explicit Q=K=V, d_k=d_v=2; K transpose; QK transpose; sqrt(2) scaling | Correct dimensions, exact integer products and displayed three-decimal scaled scores |
| 2 | Unmasked row softmax, multiplication by V, additive causal mask | Method correct; small rounding/truncation inconsistencies in probabilities and output values |
| 3 | Masked score matrix, causal softmax, output, explanation of token visibility | Correct mask and visibility. Last-row probability uses .504 while page 2 uses .503; exact value .50348984 rounds to .503 at three decimal places |
| 4 | Continuation explaining next-token prediction without future information | Correct explanation; “causal” is misspelled “casual” on page 3, with no change to mathematical meaning |

No structural attention or masking error was found. Use the original together
with the corrected numbers below. The owner should be ready to explain the
derivation orally; computational verification cannot establish individual mastery.

## Exact construction and intermediate results

Let Q=K=V, each shape (3,2), and d_k=d_v=2:

```text
Q = K = V = [[1,0], [0,1], [1,1]]
K^T       = [[1,0,1], [0,1,1]]
QK^T      = [[1,0,1], [0,1,1], [1,1,2]]
S=QK^T/sqrt(2)
          = [[.70710678,0,.70710678],
             [0,.70710678,.70710678],
             [.70710678,.70710678,1.41421356]]
```

For each row, subtract its largest permitted score, exponentiate, and divide by
the sum of those exponentials. For example, unmasked row 1 has denominator
`2 exp(1/sqrt(2)) + 1`; causal row 2 has denominator `1 + exp(1/sqrt(2))`.
The stable subtraction gives identical ratios without changing the mathematics.

```text
softmax(S) = [[.40111209,.19777581,.40111209],
              [.19777581,.40111209,.40111209],
              [.24825508,.24825508,.50348984]]
softmax(S)V = [[.80222419,.59888791],
               [.59888791,.80222419],
               [.75174492,.75174492]]

M = [[0,-inf,-inf], [0,0,-inf], [0,0,0]]
S+M = [[.70710678,-inf,-inf],
       [0,.70710678,-inf],
       [.70710678,.70710678,1.41421356]]
softmax(S+M) = [[1,0,0],
                [.33023845,.66976155,0],
                [.24825508,.24825508,.50348984]]
softmax(S+M)V = [[1,0],
                 [.33023845,.66976155],
                 [.75174492,.75174492]]
```

For example, the final output component is
`.24825508*1 + .24825508*0 + .50348984*1 = .75174492`.
The original unmasked .197 probability should round to .198, and its .598 output
should round to .599 at three decimals; causal
row 2 .330/.670 is correct at that precision. The original .752 causal final
output is the correctly rounded exact value, while page 2 .751 is truncation.
Rows sum to one at full precision; apparent .999 sums in handwriting are rounding.

Token 1 sees only token 1. Token 2 sees tokens 1 and 2. Token 3 sees all three.
The diagonal is allowed: an input token at position i predicts target i+1,
so seeing the current input does not reveal that next target.

## Independent verification

[The builder](../../../tools/assignment_evidence.py) evaluates each dot product,
stable softmax and weighted sum using scalar Python loops and `math.exp`, then
compares both causal and bidirectional outputs against the actual frozen
`scaled_attention` implementation in float64 at absolute tolerance 1e-14.
[The receipt](../verification/assignment_audit.json) preserves full-precision
matrices. The generated [companion figure](figure_01_attention_derivation.png)
uses those checked values. This is a computational cross-check of a manually
explainable derivation, not a claim that calling a library constitutes hand work.
