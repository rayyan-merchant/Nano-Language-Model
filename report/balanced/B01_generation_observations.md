# B01 genuine fixed-grid observations

The entire original100-slot grid was read, including every sampled continuation,
not only favorable examples. Slot indices below are zero-based positions in
`generation.json`; settings and checkpoint hashes accompany each original sample.
No generated text was repaired. These qualitative observations are not blinded
human scores or statistical estimates. Automated full-grid descriptors and
train-only exact overlaps are provided separately in the comparison files.

## Best checkpoint: update3,000

All five greedy outputs develop repeated phrases or dialogue loops. Slot0 begins
with the malformed “a brid.” and repeatedly alternates variants of “I know thou,
sir, I'll tell thee what I have done.” Slot10 repeats JULIET's “I know not what I
have done to do it.” Slot20 repeats the same construction and “I'll not be a
man.” Slot30 repeats “Alas, I am not so.” and then LUCIO's “I know not, sir.”
Slot40 repeatedly invokes the king of York and contradictory king declarations.
These are genuine failures of sustained dialogue continuity despite recognizable
speaker labels and frequent well-formed individual words.

Temperature0.7 usually disrupts those long greedy loops but does not make the
samples grammatical or causally coherent. Slot1 invents “orpoison”, “downpiring”
and “noddeness”; slot11 moves ROMEO through several unrelated plays' characters;
slot23 contains a more recognizable ROMEO/JULIET exchange but still changes
subject abruptly and uses broken constructions. Slot33 retains the citizen/
CORIOLANUS theme briefly before switching speakers and repeating “sir”. Short
locally plausible clauses coexist with malformed morphology and scene drift.

At temperatures1.0/1.2, novelty also includes spelling damage, merged labels and
syntactically broken phrases. Examples include “Gentle BERBERROMEO” (slot19),
“Second SimISABELLA” (slot37), “ABHORP OF EPISHIAShalf” (slot49) and trailing
“unCORIOLANUS” fragments. Several prompts with the same sampling seed share
openers such as “How would I advance you…” or “Anon should we'll…”. Fixed seeds
deliberately couple the sampling RNG across prompts; this recurring phrasing
shows limited prompt-conditioned differentiation, not proof of copied output.

## Final stopping checkpoint: update4,500

The later checkpoint does not eliminate repetition. Greedy slot70 enters a long
Nurse “O, help!” loop; slot50 repeats ISABELLA's “I am a brother”/“I am a man”;
slot60 repeats JULIET's “Nurse, I am a bawd”; slots80/90 repeat citizen or royal
phrases. A lower repetition descriptor for some final outputs would not establish
better grammar or coherent dialogue. The best and final weights are different,
so their outputs must remain separately labeled.

Final temperature0.7 samples often contain shorter recognizable exchanges: slot63
starts with ROMEO/MERCUTIO/BENVOLIO, and slot73 with JULIET/LADY CAPULET/Nurse.
They still have poor agreement, inconsistent roles and later scene drift. Slot51
mixes FLORIZEL, ANGELO, LARTIUS and the Provost; slot93 resembles historical verse
but invents “forcend” and loses a stable narrative. Higher-temperature outputs
include malformed labels (“LORD WILLOUGHBUGNE”, slot54; “GentlemanERBY”, slot69;
“MORID SRESCALUS”, slot89) and incoherent blends. None demonstrates reliable
long-range understanding or usable general language competence.

## Byte validity, presentation and interpretation

Raw token-byte diagnostics find five invalid UTF8 outputs: best slots5,8,18,48
and final slot78. The original decoder's replacement characters remain visible.
Their token IDs and strict byte-error receipts are retained; no Unicode repair
was applied. Invalid byte output is a model/tokenization-decoding limitation,
separate from the local Windows console's cp1252 display exception, which was
resolved by explicitly setting stdoutUTF8 during read-only inspection.

For a bounded demonstration, temperature0.7 is preferable to these particular
greedy loops and the visibly noisier1.2 samples. This is a recommendation from
the unchanged fixed grid, not a newly tuned sampling experiment, guarantee of
quality, or reason to suppress bad outputs. Use the selected validation-best
checkpoint for the predeclared selection policy even when individual final
samples seem locally smoother. Keep the full grid accessible next to any excerpt.

These observations support separating likelihood, compactness and textual
quality. B01's competitive validation score did not solve F01/E02's broader
generation limitations. Training-text overlaps are indicators needing context:
ordinary Shakespeare phrases recur, and exact overlap alone does not prove a
particular memorization mechanism or an absence of generalization.
