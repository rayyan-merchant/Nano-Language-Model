# E04 seed42: reviewed interpretation, negative findings and recovery record

## Observations

E04 completed5000 actual updates with all20 full sliding validations and no
research failure/resume events. Validation decreased at every scheduled point;
best and final are both5000. The first recorded training loss was4.33199036
at update1; first validation was2.49584552 at250. No step-zero measurement exists.
Final training loss1.48602149, last100-update mean1.48525713, validation1.55111692,
conditional BPB2.23778869 and character perplexity4.71673547 are actual records.

E04 is above E01 and E03 at every matched validation checkpoint. Final loss is
0.01510482 nats/token (+0.9834%) above E01 final; best loss is0.01644891 (+1.0718%)
above E01 best. E04's best/final BPB exceeds E01 best by0.02373076 and E03 best
by0.16451793. Against E03, final/best NLL is higher by0.11403514 (+7.9352%).
Loss and BPB are mathematically linked for this matched character-byte evaluation;
they are not independent pieces of evidence. Twenty correlated checkpoints are
not20 experimental replications. No significance test or seed variance is available.

Training-only E04 runtime1001.78457s is9.2806% above E01 and9.5635% below E03.
Full-validation time413.79723s is1.6894% below E01 and7.5345% below E03.
Peak allocated159475200bytes (152.0874MiB) is393216bytes (0.375MiB) above both.
The parameter increase is24576 (+1.3617%). These separate sessions of the same
T4/profile are not repeated controlled timing benchmarks; small timing differences
cannot isolate a positional-method speed effect. Two GPUs were attached, GPU0
alone trained. Driver580.178.04 advertises CUDA13 support; the actual PyTorch build
and runtime are2.5.1+cu124/CUDA12.4, not a silently upgraded research environment.

## Learning dynamics and generalization

Per1000-update validation gains diminish:0.239266 (1k–2k),0.126606 (2k–3k),
0.079796 (3k–4k),0.046598 (4k–5k). Total3k–5k improvement is0.12639416,
greater than E01's0.10344149 and E03's0.06268260 over those same intervals,
but E04 starts that interval at a worse loss. Larger remaining improvement is
not equivalent to better final performance. The descriptive late least-squares
slope is−0.06382621 nats/token per1000updates. Tiny improvements from3750 to4000
and4750 to5000 do not establish a sustained plateau: intervening evaluations
improve appreciably. Conversely, best at5000 does not prove future improvement.

There is no scheduled validation deterioration or observed double descent.
Final validation minus the last100 training mean is0.06585979. Training samples
random windows with dropout, whereas validation exhaustively scores a fixed suffix
without dropout; this gap is not an equal-protocol corpus generalization estimate.
Five updates had a preclip norm>1 (0.1%); LR stayed.0003, accumulation8, precision
float32, and no AMP updates were skipped. All20 optimizer/scheduler counters and
history lengths independently match their checkpoint step numbers.

## Learned-table measurements and attribution limits

All128 position vectors changed from the unmodified canonical seed42 initialization.
Final/best mean vector norm is13.05181885; mean L2 change is1.61847425.
[Measured row profiles](figures/position_norms.png) and
[machine-readable table diagnostics](evidence/learned_table.json) are descriptive.
They do not demonstrate learned syntax, useful position specialization or the
cause of the loss difference. Position indices describe each cropped local context,
not absolute document offsets. Full sliding scoring mostly predicts at the last
context index, while training scores all positions; this unchanged evaluation
policy must accompany interpretation of absolute-position behavior.

**Interpretation:** The expected extra table is active, trainable and changes
during optimization. It did not improve seed42 prediction at5k despite greater
capacity. Learned-table initialization consumes extra RNG draws before later
layers, so same-seed later-layer initial weights differ from E01/E03. Added capacity,
initialization and positional behavior are not independently isolated here.
**Hypothesis, not a result:** More optimization or other seeds might change the
ranking. No10k crossing, asymptotic convergence or length extrapolation is predicted.

## Actual generation failures and qualitative comparison

The full protocol yields100 recorded rows:50 settings for each checkpoint policy.
Best/final coincide, and all50 pairs match exactly. These duplicate policy labels
are retained faithfully rather than presented as independent models.

All five E04 greedy continuations loop around senators, season, seas and send,
with malformed forms such as "seast", "sease" and "senself". Repeated
eight-character-position fractions, in fixed prompt order, are0.727,0.635,0.598,
0.715 and0.655. E01's matching values are0.534,0.410,0.490,0.574 and0.606:
E04 is more repetitive on all five prompts. Versus E03 it is less repetitive
on Romeo, Juliet and king, but more repetitive on the first and citizen prompts.
This mixed descriptive diagnostic supplies no overall quality ranking.

The predetermined temperature.7/seed42 continuations use speaker labels and blank
lines, but contain "I cannot not", "my have heart", "grue", "affice", "discate"
and "sentle". Multiple prompts converge toward senators/distress/love-heart
language, showing weak distinct prompt conditioning and no reliable narrative
consistency. Temperature1/1.2 introduces more varied words and speakers, alongside
malformed strings such as "indeediencies", "princeluce" and "DUKE VIONCENTIO".
Lower repetition at higher temperature does not establish better grammar or
semantic coherence. All settings remain unedited; no favorable-only selection.

Train-only overlap checks on ten predeclared best greedy/temperature.7seed42
continuations find longest exact substrings13–25 characters. Ordinary phrases
and speaker labels are compatible with either learning or memorization; this
limited check neither proves long-passage memorization nor excludes it elsewhere.
No test-text likelihood or selection was performed. No replacement character
appears in the50 unique best-setting records. E02's separate byte-token decoding
failure and longer256-token text outputs are not directly comparable quality
measurements. E02 appears only in the contextual study BPB table: vocabulary,
capacity, byte context and49991931 target-byte exposure differ.

## Conclusion and recommendation

For this controlled configuration and seed at5000 updates, learned absolute
positions provide no predictive advantage over E01 or E03, add parameters/memory,
and retain substantial generation failure modes. This is a negative single-seed
observation, not proof that learned positions are universally worse.

Keep5k as initial screening; prioritize separately approved additional seeds and
selected10k continuations for convergence questions. E04's continuing late gains
make a continuation potentially informative, but do not justify asserting that
it will catch E01/E03 or increasing every configuration's budget. Preserve exact
optimizer/scheduler/RNG/sampling lineage if any extension is approved. No other
run, seed or extension was launched, and E01/E02/E03 evidence was not modified.
The broader three-seed study remains separately gated.

## Publication, recovery, failures and measured boundaries

Original and independent fresh recovery notebooks both published COMPLETE v1:

- https://www.kaggle.com/code/rayyannmerchant/nano-e04-pilot-20261012-5e7b13ef
- https://www.kaggle.com/code/rayyannmerchant/nano-e04-recovery-20261012-d31e262c

Archive407744239bytes SHA256:
`60b25eeab43606cedf1ef1e4ab86f9dfbbe10ccc366c1bac290015996e3a86fe`.
All58 files passed CRC, safe-path, complete file-set and per-file SHA checks.
All20 checkpoints and21 snapshot manifests were verified. Best/final5000 checkpoint
SHA256:`c2041f88a6643bc6330c2fcf60b07216f7138d87037671ef5d62ca0d4878b317`.
Independent CPU inference and fresh matching CUDA model/optimizer/scheduler/disabled
scaler/RNG/sampling-state roundtrip passed with exactly zero additional updates.
This is not a claim that a10k continuation or AMP research test was executed.

Authoritative local recovery:
`D:/Nano-Language-Model-artifacts/E04_published_20261012_5e7b13ef/verified`.
Checkpoints are under
`runs/E04_seed42_f3121bbd190e44f984afbd5dfaea57a0/checkpoints`.
The complete archive and manifest are one level above verified. D storage was
reserved before download because C had limited capacity. No E04 extraction or
publication failure occurred, and no previous artifact was deleted.

During final small-report packaging, a transient C-drive-full tool failure
prevented process startup and truncated .gitattributes. Once space became
available, that file was restored exactly from the unchanged committed version
before adding only E04 preservation rules. Previous research files, recovered
archives and scientific source were unaffected. This local delivery incident
occurred after successful publication and independent CPU/GPU recovery.

Local read-only monitoring SSE connections repeatedly ended prematurely and were
reconnected. This did not restart or interrupt training; the authoritative5000-step
journal records zero failure/resume events. A preflight CLI quota request used an
unsupported output-format option; the documented `kaggle quota` command succeeded
with28.68GPU hours remaining. These client issues are not hidden training failures.

Measured original clone-to-archive pipeline1766.15717s excludes queue, notebook
conversion/publication and retrieval. Generation100rows188.61067s and archive
compression/hash22.74731s lie within it and must not be added again. Checkpoint
calls2.68210s, verified snapshot calls7.06606s and journal calls20.08341s are separately
measured. CLI wall1455.08598s includes other bookkeeping/setup. Independent download
scope126.90599s includes API/auth/download. CPU archive verification/inference
33.17414s excludes download. Fresh GPU recovery has no terminal pipeline timing
marker, so exact duration is unavailable; log span is not a measured runtime.
Publication overhead, dollar cost and FLOPs were not measured.

## Actual tests, commands and reproducibility

Windows `python -m pytest -q --basetemp D:/Nano-Language-Model-artifacts/E04_qualification_20261012/pytest`
passed38 tests, one CUDA skip,145.75s. Training commit CPU CI38199142569 succeeded
before submission. Thirteen independent learned-position checks passed on CPU
and actual CUDA. One real-data CUDA development update verified forward/backward,
checkpoint reload/export and Agg plotting/short generation before independently
reseeded research. Those development measurements are excluded from research.
No full38-test suite rerun on Kaggle is claimed for this experiment.

The exact research CLI arguments are in evidence/execution_command.json. Executed
builder commands pinned b260264521ceb280f356b82ecd8cf4b69cb8af21 with `--experiment E04`;
recovery adds `--source-kernel rayyannmerchant/nano-e04-pilot-20261012-5e7b13ef/1 --restore-only`.
Original source/package core and qualified freeze passed guards in both runtimes.
Engineering source/diff/tests review is documented; no human approval is claimed.
PR8 remains reviewable, not merged by this task.

Reproduce retrieval into a new empty directory on the legitimate account:

```powershell
kaggle kernels output rayyannmerchant/nano-e04-pilot-20261012-5e7b13ef/1 -p NEW_DOWNLOAD_DIRECTORY --file-pattern '(E04_seed42_pilot.zip|artifact_manifest.json)$'
python scripts/pilot/inspect_archive.py NEW_DOWNLOAD_DIRECTORY/E04_seed42_pilot.zip NEW_DOWNLOAD_DIRECTORY/artifact_manifest.json NEW_EMPTY_RECOVERY_DIRECTORY
python scripts/pilot/compare_learned.py --baseline E01_VERIFIED_ROOT --rope E03_VERIFIED_ROOT --learned E04_VERIFIED_ROOT --root . --output NEW_COMPARISON_DIRECTORY
python scripts/pilot/diagnostics.py --artifacts E04_VERIFIED_ROOT --training-text CANONICAL_TRAIN_TXT --output NEW_DIAGNOSTICS_JSON
python scripts/pilot/learned_table_diagnostics.py --artifacts E04_VERIFIED_ROOT --output NEW_TABLE_DIAGNOSTICS_DIRECTORY
```

`scripts/pilot/write_e04_report.py` regenerates tables, measured summaries and
figures using the verified archive, comparison, platform/recovery receipts,
diagnostics, execution review and unchanged protocol. The written interpretation
is reviewed prose; its referenced quantities are retained in machine-readable
summary/comparison/diagnostic files. Only lightweight evidence/plots are committed;
large checkpoints, complete journals and datasets remain outside GitHub.
