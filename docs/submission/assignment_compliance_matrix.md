# Original assignment compliance matrix

Audit baseline: reviewed main `ee2b941311cfb451a73dbc65207ad3dfabb5977f`,
2026-10-15. PR [24](https://github.com/rayyan-merchant/Nano-Language-Model/pull/24)
was merged at 2026-10-15 14:10:56 UTC; this work is based on that merge, not an
unmerged dependency. No research or selected-checkpoint definitions were changed.

## Current delivery update — 2026-10-17

The [published Medium article](https://medium.com/@negation/we-built-and-deployed-a-gpt-from-scratch-the-scores-improved-the-writing-didnt-always-follow-feaf0b8943ca) is publicly readable under Rayyan Merchant's name.
Publication and section coverage were checked in the browser; the original October
15 technical receipts were not rewritten. [Delivery verification](verification/delivery_status_20261017.json)
records the check and its limits. The owner reports both recording and presentation
delivered; evidence links are pending. PARTIAL for those items means delivery was
reported but has not been independently checked, not that it must be produced again.
Publication is distinct from course submission and instructor acceptance.

## Authority and method

All **11 pages** of the supplied `A1_AIE.pdf` were read, including Sections 1–20.
Handout SHA256: `30e543d6d429f60bba96f7c20e2131d198551112b018520f6eae6dd8a23c6bdd`.
The complete instructor handout is not republished here. Page numbers below refer
to its PDF pages, including the cover. Requirement cells identify clauses in concise
wording. [Verbatim source anchors for all 103 row IDs](verification/handout_requirement_quotes.json)
were checked against the original PDF's extracted text after whitespace normalization;
use each row ID to locate its exact handout clause. Compound rows summarize related
requirements. Neither the matrix nor those excerpts replace the original handout.
The historical [compliance document](../assignment_compliance.md) was used only
as a navigation aid. Its old pending states were not treated as current results.

**PASS** means inspected content or execution evidence demonstrates the bounded
requirement, not that grading or human authorship is certified. **PARTIAL** means
some evidence exists but a literal requirement or quality dimension is unresolved.
**MISSING** means no qualifying artifact was found. **BLOCKED** means a stated
external decision prevents closure. **PENDING HUMAN DELIVERY** means an owner or
instructor action, oral explanation, publication or presentation remains.
“None” in corrective action means no technical correction identified within scope.

Additional delivery evidence: **M** = [published Medium article](https://medium.com/@negation/we-built-and-deployed-a-gpt-from-scratch-the-scores-improved-the-writing-didnt-always-follow-feaf0b8943ca); **D** = [October 17 delivery receipt](verification/delivery_status_20261017.json).

Abbreviations below link to real inspected evidence:

- [F1–F10](evidence/README.md): figure index, captions, exact sources and caveats.
- [C](evidence/concepts_and_objectives.md): all seven concepts, MLM/CLM, complexity.
- [A](verification/assignment_audit.json): current scalar math, shapes, receipt consistency and source hashes.
- [R](reproducibility_report.md): prior executed fresh-environment report with raw receipt links.
- [L](verification/colab-v3/full-tests.txt): complete clean Linux test log.
- [N](verification/colab-final/NanoLM_Submission_Demo_executed.ipynb): actual final executed notebook.
- [G](verification/colab-final/receipts/generation/receipt.json): unedited generation/token IDs.
- [P](../../report/final/E01_E14_consolidated_analysis.md): genuine all-fourteen analysis.
- [Q](../../report/final/F01_10000_analysis.md), [B](../../report/balanced/B01_model_selection.md): accepted combined-run reports.

## Sections 1–4: mission, concepts and deliverables

| ID / PDF page | Handout requirement | Evidence expected | Actual evidence | Verification method | Status | Corrective action |
|---|---|---|---|---|---|---|
| 1 / 2 | Build the raw-text to generated-text causal LM pipeline | Connected implementation and execution | [Model](../../src/nano_lm/model.py), [training](../../src/nano_lm/train.py), N, G | Inspected own decoder and actual model/generation receipts | PASS | None |
| 2a / 2 | Explain sequential versus parallel computation | Reasoned or measured comparison | C §1 | Checked training vs autoregressive distinction and work/memory qualifications | PASS | Owner explain orally |
| 2b / 2 | Implement scaled self-attention and causal masking | Math, code, tests | F1/F2, [attention](../../src/nano_lm/attention.py), A | Independent scalar-loop comparison, normalization and exact zero future weights | PASS | None |
| 2c / 2 | Implement multi-head attention with correct tensor shapes | Split/merge trace | F3, A | Actual forward hooks plus independent score/value reconstruction | PASS | None |
| 2d / 2 | Explain positions and Transformer components | Positional visualization and decoder explanation | F4/F5, C | Actual sinusoidal output, parameter construction, pre-LN source inspection | PASS | None |
| 2e / 2 | Contrast BERT MLM and GPT CLM | Objectives, visibility and constructed examples | C, MLM/CLM figure | Same QKV evaluated with causal/noncausal masks, shifted-target explanation | PASS | No full BERT training required |
| 2f / 2 | Create shifted inputs/targets and token CE | Batching and loss tests | [Core tests](../../tests/test_core.py), L | Inspected `test_split_and_shift`, manual CE comparison and executed pass log | PASS | None |
| 2g / 2 | Train/evaluate own LM without pretrained model | Genuine training records, explicit code | P/Q/B, N | Compared summaries, source and selected model identities | PASS | None |
| 2h / 2 | Run controlled ablations and expose online FastAPI inference | Measured comparisons and real API | P, F9/F10, [deployment record](../deployment/README.md) | All-14 summaries checked; saved API matches canonical generation | PASS | Hosting availability is time-dependent; recheck before live delivery |
| 3a / 3 | Why Transformers: depth/speed/memory | Hand analysis and reasoned comparison | C §1 | Asymptotic work, sequential depth and generation limitations reviewed | PASS | None |
| 3b / 3 | Self-attention: Q/K/V, scaling, softmax, mask | Worked numeric example, implementation, unit test | F1/F2, A, L | Exact 3×2 calculation and frozen implementation agree | PASS | None |
| 3c / 3 | Multi-head: reshape/split/merge and head outputs | Shape trace/output | F3, A | Real E02-config forward trace; merge and projected output agree | PASS | None |
| 3d / 3 | Positional encoding | Visualization and explanation | F4, C | Actual sine/cosine values and frequency variation checked | PASS | None |
| 3e / 3 | Full Transformer: FFN, LayerNorm, residual/block | Code, diagram, shape checks | F5, C, [model](../../src/nano_lm/model.py) | Actual construction/counts and pre-LN equations inspected | PASS | None |
| 3f / 3 | BERT-style MLM versus GPT-style CLM | Mask matrix and short comparison experiment | C, A | Bidirectional vs triangular numerical example, targets and objective distinctions | PASS | Illustration is not trained BERT evidence |
| 3g / 3 | GPT-style causal LM | Shifted targets, loss curves, autoregressive generation | F6/F8/F10, core tests, N | Genuine results plus test receipts, no inferred new measurements | PASS | None |
| 4a / 3 | Public or instructor-accessible repository | Independent repository access | [GitHub](https://github.com/rayyan-merchant/Nano-Language-Model), PR24 merge | Anonymous GitHub API status and reviewed main fetched | PASS | Supply final reviewed submission revision |
| 4b / 3 | Google Colab script | Executable independent demo | [Guide](colab_demo.md), N, R | 13 final executed cells, no errors; earlier transfer/receipts checked | PASS | Instructor uses own Google account |
| 4c / 3 | Trained checkpoint or command loading from GitHub/shareable Drive | Accessible trained artifacts | [Model access](model_access.md), [manifest](models.json), public HF | Anonymous immutable HF metadata and six-file receipts match hashes | PARTIAL | Ask instructor to accept HF instead of named GitHub/Drive; do not publish a mirror without approval |
| 4d / 3 | README: setup, dataset, training, inference, expected output, limitations | Commands and usable documentation | [README](../../README.md), R, [submission guide](README.md) | Inspected commands against actual CLI and installed receipts | PASS | No new training authorized merely by documenting commands |
| 4e / 3 | Medium technical report, recommended 8–12 pages | Submitted final integrated article | M/D; public Medium article by Rayyan Merchant | Browser verified public article and ten main sections | PASS | Retain course-submission receipt; instructor/author review is not inferred from publication |
| 4f / 3 | At least six evidence items; Section 12 lists ten | Complete evidence index | F1–F10 | All ten content checks and visual review | PASS | Confirm all required evidence is represented in the published article and linked submission package |
| 4g / 3 | Ablation table and discussion | At least three controlled comparisons | F9/P | Depth, heads, context and all-14 table checked against summaries | PASS | Preserve confound and single-seed caveats |
| 4h / 3 | 5–10 minute recorded demonstration | Actual recording | D; owner reports recording delivered | No recording URL supplied; duration/content/access unverified | PARTIAL | Provide existing recording link; verify 5–7 minutes and required walkthrough items |
| 4i / 3 | Class presentations | Delivered presentation and individual explanation | D; owner reports class presentation delivered | No independent delivery acknowledgement supplied | PARTIAL | Attach presentation evidence; do not infer individual oral mastery |
| 4j / 3 | Clone/install/tests/generate without undocumented setup | Clean-environment acceptance | R, L, Windows install/restart receipts | Independent receipt inspection, exact selected package identities | PASS | Use documented short Windows temp path |

## Sections 5–8: milestones, handwritten work, implementation and constraints

| ID / PDF page | Handout requirement | Evidence expected | Actual evidence | Verification method | Status | Corrective action |
|---|---|---|---|---|---|---|
| 5.M1 / 4 | Numeric attention, shapes and future blocking | Manual and executable demonstration | F1/F2, A, core tests | QK/scaling/softmax/V recomputed independently; future probability zero | PASS | Read disclosed rounding corrections |
| 5.M2 / 4 | Independent heads, concatenation/output and sinusoidal positions | Shapes, code, visualization | F3/F4 | Actual head/output tensors and sinusoidal construction checked | PASS | RoPE alone would not suffice; sinusoidal explicitly included |
| 5.M3 / 4 | Full decoder and MLM contrast | Integrated block and comparison | F5/C | Pre-LN source, complete objective demonstration | PASS | None |
| 5.M4 / 4 | Fixed contexts, shifted targets, CE, tiny overfit, loss/LR/step/sample logs | Passing small test and training logs | L, core `test_tiny_overfit`, P/Q/B, G | Existing 60-update fixture must end below .15 initial loss; executed log includes pass | PASS | Synthetic test is not Shakespeare research evidence |
| 5.M5 / 4 | Real corpus run, parameters/config/hardware/time/final losses/five prompts | Actual research records | P/Q/B, summaries, F8/G | Real summaries and deterministic five-prompt selection inspected | PASS | Retained time differs from physical replay cost |
| 5.M6 / 5 | Ablations, CLI/API hosting, report | Measured comparisons, deployed demo and final report | P/F9/F10, R, M/D | Ablations and API evidence retained; final article now public | PASS | Recording/presentation evidence and course acceptance remain separately tracked |
| 6a / 5 | Three tokens, d_k=d_v=2, explicit Q/K/V | Handwritten matrices | Original PDF p1, F1 companion | All four original pages visually inspected | PASS | None |
| 6b / 5 | QK transpose and sqrt(d_k) scaling | Intermediate arithmetic | Original p1, corrected companion | Exact integer products and float64 scalar reference | PASS | None |
| 6c / 5 | Row softmax and multiplication by V | Normalized weights and output | Original p2–3, companion | Rounding/truncation discrepancies documented, corrected full-precision values checked | PASS | Submit original with companion; do not conceal correction |
| 6d / 5 | Causal masked score matrix and token visibility | −infinity future scores, explanation | Original p2–4, F1/F2 | Correct diagonal/past mask and information-availability prose | PASS | None |
| 6e / 5 | Show personal manual understanding before relying on PyTorch | Student explanation/hand work | Historical handwritten original plus typed audit | Original exists; timing and personal mastery cannot be certified by automated checks | PENDING HUMAN DELIVERY | Each author explain calculations orally; typed companion is not a replacement for personal hand work |
| 7a / 5 | Deterministic tokenizer/vocabulary/special handling | Roundtrip/serialization and fitting provenance | [BPE](../../src/nano_lm/bpe.py), [tokenizer](../../src/nano_lm/tokenizer.py), assets, L | Train-only fitting, explicit IDs, checksum/roundtrip tests and package receipts inspected | PASS | Explain byte BPE and Unicode replacement limitations |
| 7b / 5 | Embeddings and own sinusoidal encoding | Actual code and values | F4/F5, core position tests | Actual sinusoidal function used; no pretrained position module | PASS | None |
| 7c / 5 | Explicit Q/K/V, scaled scores, softmax, optional mask | Own attention implementation | F1/F2, attention source, A | Scalar reference and causal/noncausal comparison | PASS | None |
| 7d / 5 | Configurable heads and split/merge | Real config shapes | F3, core locked-registry tests | E02 trace and registry dimensions inspected | PASS | None |
| 7e / 5 | Two-layer FFN with nonlinearity, LayerNorm, residuals | Decoder source and shape checks | F5/C/model, core tests | GELU, two projections and pre-LN residual equations checked | PASS | None |
| 7f / 5 | Future exclusion and vocabulary logits | Causal mask, LM head | F2/F5/F7 | Trained future probabilities exactly zero, head shape verified | PASS | None |
| 7g / 5 | Shifted token CE and autoregressive generation | Loss/target/generation tests | Core tests, L, G/F10 | Test assertions inspected and actual outputs/IDs checked | PASS | None |
| 7h / 5 | Temperature and maximum generation length | Implemented decoding controls | [Generator](../../src/nano_lm/generate.py), API schemas, F10 | Real request validates; recorded settings/output agree | PASS | Top-k/top-p are optional, not required or claimed |
| 8a / 5–6 | No pretrained GPT/model weights or black-box Transformer architecture | Own source and traceable trained weights | Scientific source, model manifest | Explicit tensor attention, nn.Linear/LayerNorm/GELU; own-checkpoint lineage | PASS | None |
| 8b / 6 | Basic tensors/autograd/optimizers and tokenizer libraries allowed | Explain library boundaries | C, source and dependencies | No AutoModel/GPT2 or nn.Transformer architecture in scientific source | PASS | Explain tokenizer library use in final article |
| 8c / 6 | Cite external learning/code and explain adaptations | Attribution and individual understanding | C references; research reports | New diagrams/equations attributed; exhaustive historical-source audit not established | PARTIAL | Authors review all adapted ideas/snippets and citations before final delivery |
| 8d / 6 | Clean-checkout reproducibility | Commands/install/tests/inference | R/L/N/restart | Actual clean receipts inspected, not filename-only assertions | PASS | None |
| 8e / 6 | No secrets/private corpus/unnecessary large artifacts; gitignore/env | Git content and checks | Current scoped secret/size checks; integrity receipt | New tracked files inspected; models/datasets remain external | PASS | Pattern checks are not proof that no historical secret ever existed |

## Sections 9–13: corpus, organization, configuration, figures and experiments

| ID / PDF page | Handout requirement | Evidence expected | Actual evidence | Verification method | Status | Corrective action |
|---|---|---|---|---|---|---|
| 9 / 6 | Choose suitable corpus; Tiny Shakespeare recommended | Source/size/split explanation | [Data guide](../../data/README.md), research protocol, Q/B/P | Pinned source, 90/5/5 byte boundaries and train-only vocab provenance inspected | PASS | Preserve usage/source attribution; no private corpus substituted |
| 10 / 6–7 | Clear model/data/tests/experiments/report locations | Navigable repository and README | [Project map](../README.md), root README | Existing supported paths and new index checked for working links | PASS | Suggested handout tree is not mandatory |
| 11a / 7 | Vocabulary, context, width, heads, layers, FFN configurable | Actual config interface | [ModelConfig](../../src/nano_lm/config.py), frozen configs, F3/F5/P | Constructor and locked-registry tests inspected; actual models instantiate | PASS | None |
| 11b / 7 | Dropout, optimizer, learning rate, batch size configurable | CLI/config interface and saved settings | [Commands](../commands.md), train source, configs, P | Documented argparse options and resolved research records agree | PASS | New executions still require authorization |
| 11c / 7 | Temperature configurable; justify starting settings | Decoding interface and rationale | F10, API schemas, Q/B/P | Request parsed by actual schema; real settings recorded | PASS | Suggested 128/256 widths etc. are guidance, not compulsory values |
| 12.1 / 7 | Hand-calculated attention | Complete numeric and causal derivation | F1 | Original + corrected companion independently checked | PASS | Human understanding remains separate |
| 12.2 / 7 | Causal mask visualization | Clearly labelled allowed/future positions | F2 | Actual mask convention verified | PASS | None |
| 12.3 / 7 | Multi-head shape trace | Projections/splits/weights/merge/output | F3 | Actual tensors, reconstruction, output comparison | PASS | None |
| 12.4 / 7 | Positional encoding visualization | Sinusoidal patterns and explanation | F4 | Actual function output, dimension frequencies | PASS | None |
| 12.5 / 7 | Model architecture and parameters | Actual construction/summary | F5/A | All three constructed counts equal manifest | PASS | None |
| 12.6 / 8 | Train/validation curves | Genuine logged observations | F6 | 40 F01 validation rows, minimum and final agree with summary; image inspected | PASS | Joined line segments are not extra observed evaluations |
| 12.7 / 8 | Trained attention heatmap | Genuine checkpoint attention | F7 | B01 step3000, 10×10 recorded probabilities, source identities, image inspected | PASS | One head is not a universal interpretability claim |
| 12.8 / 8 | Generated text examples | Fixed prompts and actual imperfect outputs | F8/G | All five B01 greedy cases selected in receipt order, no quality filtering | PASS | Report loops and malformed words honestly |
| 12.9 / 8 | Ablation comparison | Measured table/figure and interpretation | F9/P | All 14 summary parameter/update/BPB values cross-checked | PASS | One seed, changing capacity/context/byte exposure |
| 12.10 / 8 | CLI/API request and response with model details | Actual inference receipt | F10 | HTTP200 saved response matches generation and schema; metadata checked | PASS | Historical local API execution, not new public uptime evidence |
| 13a / 8 | At least three controlled architectural factors | Real measured comparisons | Depth/heads/context F9, all-14 CSV | Actual intended overrides and per-family measurements inspected | PASS | Changes mechanically alter other dimensions/compute; discuss them |
| 13b / 8 | Record loss, parameters, runtime, memory/throughput where relevant | Honest comparative metrics | CSV/P plus original pilot reports | Measured values and physical/replayed timing caveats checked | PASS | Do not treat single-seed correlations as isolated causal mechanisms |
| 13c / 8 | Discuss results instead of merely listing scores | Observations vs interpretations and limitations | P/Q/B, F9 caveats | Negative outcomes, sampling/context confounds and text failures retained | PASS | No significance claim; combined models not single-factor tests |

## Sections 14–20: rubric, report, acceptance and delivery

| ID / PDF page | Handout requirement | Evidence expected | Actual evidence | Verification method | Status | Corrective action |
|---|---|---|---|---|---|---|
| 14.1 / 8–9 | Conceptual understanding (15%) | Equations, shapes, motivation, oral understanding | C/F1–5, original PDF | Technical evidence checked; personal mastery unverified | PARTIAL | Each member prepare oral explanation; no marks guaranteed |
| 14.2 / 9 | Attention implementation (15%) | Correct tested own attention | Source, A/L/F1–3/F7 | Scalar numerical, shape, mask and trained-future checks | PASS | None |
| 14.3 / 9 | Transformer architecture (15%) | Integrated correct blocks | Source/F5/core tests | Pre-LN residual, FFN, shape/count verification | PASS | None |
| 14.4 / 9 | Language-modeling pipeline (15%) | Tokenizer/data/train/eval/checkpoint | Research reports/R/N | Real training lineage and independent selected-package reload | PASS | None |
| 14.5 / 9 | Experiments and analysis (15%) | Controlled meaningful results | P/F9 | All-14 metrics and limitations verified | PASS | No independent-seed variance estimated |
| 14.6 / 9 | Generation quality and evaluation (10%) | Coherent samples plus quantitative evaluation | G/F8, Q/B diagnostics | Genuine outputs include loops, malformed spelling and incomplete coherence | PARTIAL | Discuss limits; cannot guarantee the rubric's coherence standard or full credit |
| 14.7 / 9 | Software engineering (10%) | Reproducible tested clean package | R/L/N, current CI/integrity | Existing fresh execution and current focused checks | PASS | Review new PR; technical checks do not guarantee marks |
| 14.8 / 9 | Report and demo (5%) | Clear final report, figures and demonstration | Figures, M/D; recording owner-reported delivered | Article inspected; recording content/delivery not independently checked | PARTIAL | Attach recording evidence and retain instructor assessment; no marks guaranteed |
| 15.1 / 9 | Problem statement | Final report section | M plus linked original research records | Section 1: corpus constraint, motivation and project question; browser text inspected | PASS | Section coverage verified; detailed grading, formatting and author attribution remain human review |
| 15.2 / 9 | Foundations: attention/shapes/positions/Transformer/MLM/CLM | Final report section | M plus linked original research records | Section 2: attention, shapes, positions, residual block, MLM/CLM; browser text inspected | PASS | Section coverage verified; detailed grading, formatting and author attribution remain human review |
| 15.3 / 9 | Implementation: tokenizer/config/attention/mask/train | Final report section | M plus linked original research records | Sections 2–4 and 9: tokenizer, decoder, causal targets and implementation; browser text inspected | PASS | Section coverage verified; detailed grading, formatting and author attribution remain human review |
| 15.4 / 9 | Dataset: source/size/split/vocab/context/filtering | Final report section | M plus linked original research records | Section 3: original bytes, split, vocabulary and conditioning policy; browser text inspected | PASS | Section coverage verified; detailed grading, formatting and author attribution remain human review |
| 15.5 / 9 | Setup: hardware/hyperparameters/optimizer/batch/steps/checkpoints | Final report section | M plus linked original research records | Section 4 and F01/B01 sections: T4, optimizer/LR, batching, updates and selected checkpoints; browser text inspected | PASS | Section coverage verified; detailed grading, formatting and author attribution remain human review |
| 15.6 / 9 | Results: losses/samples/counts/time/attention | Final report section | M plus linked original research records | Sections 4–7: measured plots, parameters, runtime, attention and genuine output limitations; browser text inspected | PASS | Section coverage verified; detailed grading, formatting and author attribution remain human review |
| 15.7 / 9 | Ablations | Final report section | M plus linked original research records | Section 4: controlled variant families and single-seed caveats; browser text inspected | PASS | Section coverage verified; detailed grading, formatting and author attribution remain human review |
| 15.8 / 9 | At least two failure modes and likely causes | Final report section | M plus linked original research records | Sections 5–7: generalization deterioration, repetition and malformed outputs; causal explanations qualified; browser text inspected | PASS | Section coverage verified; detailed grading, formatting and author attribution remain human review |
| 15.9 / 9 | CLI/API and reproducibility | Final report section | M plus linked original research records | Section 9: genuine API evidence, selected packages and pinned reproducibility notebook; browser text inspected | PASS | Section coverage verified; detailed grading, formatting and author attribution remain human review |
| 15.10 / 9 | Conclusion | Final report section | M plus linked original research records | Section 10: measured conclusions, limitations and unanswered questions; browser text inspected | PASS | Section coverage verified; detailed grading, formatting and author attribution remain human review |
| 16a / 9 | Fresh install and unit tests without manual edits | Clean install and complete test evidence | Windows install commands, L/R |151 Linux passes/1 CUDA skip; Windows149 +2 corrected-path passes | PASS | Do not call Windows a single clean151-pass invocation |
| 16b / 9 | Tiny-overfit loss decreases substantially | Real bounded correctness test | Core `test_tiny_overfit`, L | 60 synthetic updates, final<15% initial assertion, executed suite pass | PASS | No new research training needed |
| 16c / 9 | Reload final checkpoint in a fresh process | Genuine trained reload/generation | Restart receipts for all three, E02 terminal5000 | Hash/config/count identities and fresh-process selected models checked | PASS | B01/F01 are selected-best submission states, not terminal training states |
| 16d / 9 | Generate from three fixed prompts | Actual outputs | G (five prompts × three models × two modes) | 30 genuine cases;12 available references match raw IDs in both Colab runs | PASS | Other18 cases have no old raw-ID reference; do not claim30 reference matches |
| 16e / 9 | Future-token exclusion | Logits invariant, future weights zero | F7/A/core tests/L | Trained B01 earlier-logit error0.0, upper triangle zero | PASS | None |
| 16f / 9 | README command accuracy/config consistency/no secrets | Commands, content checks | README/R/current checks | Actual interfaces and historical executed commands checked | PASS | No blanket proof about every historic secret |
| 17.1–3 / 10 | Show repo/core file, attention/mask and passing tests | Recorded/live walkthrough | D; existing recording owner-reported delivered | Video unavailable for content inspection | PARTIAL | Provide the existing recording and check these walkthrough items |
| 17.4–6 / 10 | Show config/counts, curves, fixed generation command | Recorded/live walkthrough | F5/F6/F8/F10, D | Owner reports delivery; no recording link to verify content | PARTIAL | Inspect existing recording for counts/curves/generation and checkpoint identity |
| 17.7–9 / 10 | Interpret attention, deployed live inference, ablations/limitation | Recorded/live walkthrough | F7/F9/Studio, D | Owner reports delivery; no recording/live presentation receipt inspected | PARTIAL | Verify attention interpretation, live inference and ablation/limitation in supplied evidence |
| 18a / 10 | Each team member understands implementation and contributes | Attribution and individual oral answers | Research ownership records, D | Owner reports presentation delivered; personal understanding not independently assessable | PARTIAL | Retain contribution statement and presentation acknowledgement |
| 18b / 10 | Cite code/equations/figures and explain adaptations | Complete source attribution | C/new captions/research reports | New artifacts attributed; historical completeness remains human review | PARTIAL | Review attribution completeness in the published article |
| 18c / 10 | Allowed corpus; experimental outputs not reliable facts | Data provenance and limitations | Data guide, Q/B, Studio model limitations | Public literary corpus identified; experimental generation caveats present | PASS | Authors confirm course/data permission expectations |
| 18d / 10 | Reproducible commands; avoid large checkpoints unless requested | Small Git repository plus accessible models | model access/R/scoped size checks | Weights external, hashes pinned, no new large artifacts | PASS | HF hosting approval separately unresolved |
| 19 / 10 | Final checklist: repo, README, manual/mask/shapes/positions/architecture/curves, five samples, three ablations, counts/config, CLI/API | Accessible required evidence | F1–10/R | All listed technical checklist items inspected | PASS | Use index, not historical pending statuses |
| 19.M / 10 | Medium report submitted | Publication/submission receipt | M/D; article publication verified | No course-submission receipt supplied | PARTIAL | Retain the article submission/acknowledgement; publication alone does not establish course receipt |
| 19.S / 10 | No secrets/large unnecessary artifacts; all sources cited | Technical checks and author review | Current checks/C | Scoped checks pass; complete historical attribution remains review | PARTIAL | Complete author citation review |
| 20 / 11 | Learning resources and final principle: understandable tested implementation | Appropriate attribution and understanding | C, explicit source, tests | New educational work linked to primary papers and handout curriculum | PARTIAL | Authors demonstrate understanding; references alone are not proof |

## Overall decision

Technical evidence and public Medium publication are established within their
recorded scopes. Recording and class presentation are owner-reported delivered;
their evidence links are pending. Overall course submission, individual
understanding/attribution and instructor acceptance of HF hosting are not
independently established. No rubric score or guaranteed grade is assigned.
See [readiness and owner actions](assignment_readiness.md) for the remaining
verification actions; no new research is required by this status maintenance.
