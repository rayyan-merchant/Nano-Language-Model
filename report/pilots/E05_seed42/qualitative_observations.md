### Observation

All five predetermined greedy continuations collapse into related phrase loops: `the sense of the sense`, `the sens of the sense`, and long runs of `so so so ... soul`. The Romeo sample begins `What is the send of the sent of the sensess of the see`; Juliet changes to a CORIOLANUS turn after an already repetitive opening. The king prompt retains king/son vocabulary but does not develop a coherent proposition. These samples show character-level word patterns and line breaks, rather than reliable spelling, grammar, narrative continuity or speaker-specific relevance.

Temperature0.7 increases speaker-turn diversity and occasionally yields locally readable phrases. The fixed king/seed42 continuation contains `My lord, and his comfort.` and `Nay, we will not have me made not shall her winds:` in adjacent KING RICHARD III turns; the first phrase is locally readable while the second illustrates broken grammar. Other0.7 outputs contain `prompemised`, `entreell`, `beggginst`, and `KING RICHARD IIII`. The entire15-setting0.7 grid remains preserved.

Temperatures1.0 and1.2 produce more diverse outputs with further malformed words and names: `prompemichancel`, `incervouablest`, `Geentlestians`, `RICHAS OF SORK`, and `NORTHUMBERLE`. Shared sampling seeds can make continuations converge across prompts; this is an observed decoding behavior, not independent evidence of prompt relevance. All50 unique settings were inspected, and all100 labeled records are retained unchanged.

E01's predetermined greedy grid also loops around `the world of the world` and contains malformed forms such as `seast` and `seasure`. E05 does not show a consistent qualitative improvement over it. Mean repeated8-gram position fraction for best-checkpoint greedy decoding is0.561446 for E05 versus0.522892 for E01; this is a descriptive increase, not a language-quality score or significance test. Sampling reduces literal repeated character windows while spelling and syntax errors remain. Low repetition at higher temperature does not imply stronger coherence.

### Activation observations and interpretation

Every research gradient norm was finite;6/5000 updates exceeded the clipping threshold, versus7/5000 in E01. On the four fixed training windows,63.26–66.08% of ReLU outputs were zero across the four blocks, and **no MLP unit was inactive across all512 sampled positions**. All768 units in each block activated at least once in this diagnostic sample. These measurements offer no evidence of a permanent dying-ReLU problem at the final checkpoint, while not constituting a longitudinal all-corpus activation audit.

E05 loss is lower at the1k and1.25k measurements and higher at the other18 matched points. Best loss/BPB is1.2922% higher than E01's best; final loss/BPB is1.2036% higher than E01's final. Validation continued improving through5k, and the3k-to5k loss decrease is0.10554848 versus E01's0.10344149. There is no observed late plateau or final deterioration in E05. The measured training-only time is20.657% higher on this Colab run, with equal allocated-memory peak; uncontrolled platform/driver/host conditions prevent attributing this timing difference solely to ReLU.

## Conclusion

Under this matched seed42,5000-update baseline architecture, ReLU did not improve predictive performance or generated-text quality. The evidence supports retaining GELU as the reference activation for now. The small loss/BPB regression is an observed single-seed association, not a statistically established universal disadvantage. Additional separately authorized seeds would be needed before changing the final activation choice. No failure was established from activation sparsity, and no combined optimized configuration or extension was launched.
