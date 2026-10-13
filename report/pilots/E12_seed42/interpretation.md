# Context-length and generated-text interpretation

## Prediction, convergence and cost

E12 has worse BPB than E01 at every one of the twenty matched validation points.
The difference is largest at step 500 (about +0.22486 BPB) and narrows to +0.04028
at step 5,000. E12's best/final BPB is +0.04222093 above E01's best. It improves
from loss 1.70152845 at step 3,000 to 1.56393333 at step 5,000 and still improves
from step 4,750; the experiment stopped at the authorized endpoint. That trend
does not show whether longer training would change the ordering.

E12 takes 51.82% of E01's recorded training time (475.030 versus 916.708 seconds),
uses 1.832× its peak allocated memory (277.908 versus 151.712 MiB), and spends
2.203× as long on full validation (927.393 versus 420.908 seconds). Larger sequence
matrices increase memory while halving the number of accumulation microbatches.
That is a plausible explanation for the training speed difference, not a measured
FLOP or kernel-time attribution. The combined training-plus-validation cost is
1,402.423 seconds for E12 versus 1,337.616 seconds for E01: faster training alone
does not establish a faster research pipeline. Sessions, IO and publication differ.

Genuine reviewed E11 reaches best/final BPB 2.15396251 at step 5,000 with 116.146 MiB
allocated memory, 1,868.694 training seconds and 267.250 validation seconds. It
predicts the same suffix bytes with only 64 characters of context and 16-way
accumulation. This descriptive ranking favors shorter context for likelihood at
this seed/budget, while E11 is slower to train. It does not establish a universal
advantage of shorter context. Reference profiles and original dataset manifests
were checked; no E01/E11 retraining or fresh inference is claimed here.

## Reading the fixed generation grid

All fifty best-policy E12 settings were inspected; the fifty final-policy records
are exact paired duplicates because best equals final. Every predefined greedy
prompt is shown below, not chosen for quality. The complete unedited grid remains
in generation.json and generation.md. Sampling results include all temperatures
and seeds. E01/E11 comparisons use their preserved grids and the same protocol;
this is a descriptive inspection, not a blinded human quality benchmark.

All five E12 greedy prompts lead to repetitive phrases. ROMEO repeats “the prince
of the present of the prince”; the opening quote and The king prompts settle into
“the prince of the prince” sequences. First Citizen opens with “so be so be so”,
then switches labels without establishing a consistent conversation. JULIET also
moves to DUKE VINCENTIO while repeating prince/present patterns. Formatting,
capitalized labels and line breaks resemble plays, but neither speaker identity
nor scene continuity is reliably preserved. The initial prompt affects beginnings;
it does not produce sustained character-specific dialogue or a meaningful
continuation of the famous opening phrase.

Sampling at temperature 0.7 largely breaks the long exact greedy loops and yields
recognizable labels such as COMINIUS, ROMEO and ESCALUS. Syntax remains broken:
ROMEO seed42 begins “I cannot not this servant of death”, followed by “The gracious
are the pawn true I speak of thee”. ROMEO and JULIET at that same setting share
much of their continuation, a concrete example of weak speaker responsiveness.
Other settings contain double words (“and and”, “but but”) and malformed words
such as “disperitmes” and “purprince”. These outcomes are retained.

At temperatures 1.0 and 1.2 there is greater variation in words, punctuation and
labels but frequent spelling and grammar failures. Examples include TSecond
Citizen, Second Citizzen, DUKE BRAKENGHLOW and 3 KING HEF YORK. Temperature 1.2
can produce strings such as “dismerehmbanct” or “storophangs”. Diversity alone is
not evidence of coherent meaning. Label plausibility does not imply consistent
characters or speakers. Occasional locally plausible phrases occur amid broken
sentences; no sustained plot, correct grammar or long-context coherence is established.

E01's predefined greedy outputs also loop, often around “the world of the world”
and “the seal”; E11 repeats “the world of the world” and LADY GREY labels. E12
changes the dominant words to prince/present rather than eliminating the failure.
Its sampled output avoids many exact loops but still resembles fragmented
Shakespeare-like surface text. Lower BPB and lower repetition need not agree:
E11 has the best likelihood here and the highest greedy eight-gram repetition.

## Descriptive repetition measure

For each 256-character continuation, excluding the prompt, collect all overlapping
8-character windows. Repeated fraction is 1 minus distinct windows divided by
all windows. The table averages five greedy or fifteen sampled best-policy
records per condition; there are no standard-error or significance claims.
E12/E11 final-policy duplicates are not pooled as independent samples. This
measure catches literal reuse, but shorter repeats, semantic loops and invented
words can escape it. Low values are not grammar/coherence or memorization scores.

| Experiment | Greedy | T=0.7 | T=1.0 | T=1.2 |
|---|---:|---:|---:|---:|
|E01|0.522892|0.001339|0.000000|0.001339|
|E11|0.815261|0.003481|0.000535|0.000000|
|E12|0.604819|0.000803|0.000535|0.000268|

## All five predefined E12 best/final greedy continuations

Prompt: 'To be, or not to be'

```text
To be, or not to be so see the see
That the seem of the street of the prince,
And the prince of the present of the prince,
And the prince of the prince of the prince,
And the prince of the prince of the prince,
And the prince of the prince of the prince,
And the prover of th
```

Prompt: 'ROMEO:\n'

```text
ROMEO:
The shall be so so so death the prince,
And the seem of the present of the prince,
And the prince of the present of the prince,
And the prince of the present of the prince,
And the prince of the prince of the prince,
And the prince of the prince of the com
```

Prompt: 'JULIET:\n'

```text
JULIET:
The world the come of the prince of the part
That the seem of the present of the prince,
And the prince of the present of the present,
And the present of the come of the present.

DUKE VINCENTIO:
The world the prince of the prince of the prince,
And the pr
```

Prompt: 'First Citizen:\n'

```text
First Citizen:
The shall be so be so be so be so death.

DUKE VINCENTIO:
The see the prince of the present of the prince.

KING RICHARD II:
The come the prove of the present of the prince.

KING RICHARD II:
The see the provost of the prince of the prince,
And the seem of
```

Prompt: 'The king'

```text
The king of the prince of the prince,
And the seem of the stand of the prince,
And the prince of the present of the prince,
And the prince of the prince of the prince,
And the prince of the prince of the prince,
And the prince of the prince of the prince,
And the 
```
