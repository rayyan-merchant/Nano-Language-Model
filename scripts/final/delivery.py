"""Supplement accepted F01 reporting; reads genuine evidence and never trains."""
import argparse
import csv
import json
from pathlib import Path

from nano_lm.artifacts import atomic_json, file_hash
from scripts.pilot.diagnostics import longest_overlap


def verify_probes(output):
    """Compare genuine independent CPU and CUDA receipts; no model execution."""
    evidence = Path(output) / 'evidence/research'
    cpu = json.loads((evidence / 'independent_recovery.json').read_text(encoding='utf-8'))
    cuda = json.loads((evidence / 'CUDA_recovery.json').read_text(encoding='utf-8'))
    assert cpu['archive_sha256'] == cuda['archive_sha256']
    assert len(cpu['samples']) == len(cuda['samples']) == 2
    checks = []
    for sample in cpu['samples']:
        other = next(s for s in cuda['samples'] if s['policy'] == sample['policy'])
        assert sample['sha256'] == other['sha256'] and sample['text'] == other['text']
        assert sample['greedy'] and other['greedy'] and sample['seed'] == other['seed'] == 42
        assert sample['max_new_tokens'] == other['max_new_tokens'] == 64
        checks.append({'policy': sample['policy'], 'checkpoint_sha256': sample['sha256'],
            'greedy_text_exactly_equal': True, 'new_tokens': 64, 'seed': 42})
    atomic_json(evidence / 'recovered_CPU_CUDA_probe_equality.json', {
        'status': 'Both recovered best/final CPU and CUDA greedy probe strings match exactly',
        'archive_sha256': cpu['archive_sha256'], 'samples': checks,
        'scope': 'Comparison of genuine inference receipts only; zero model or optimizer executions'})


def supplement(root, output):
    root, output = Path(root), Path(output)
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    findings = read(output / 'findings.json')
    assert findings['fixed_generation_reproduction']['slots_exactly_reproduced'] == 100
    cpu = read(output / 'evidence/research/independent_recovery.json')
    assert cpu['authoritative_updates'] == 10000 and cpu['uninterrupted_sequential_journal']
    verify_probes(output)
    generation = read(output / 'generation.json')
    raw = (root / 'data/processed/tiny-shakespeare-v3/train.txt').read_bytes()
    assert file_hash(root / 'data/processed/tiny-shakespeare-v3/train.txt') == cpu['sampler']['train_text_sha256']
    # The independent recovery already validates canonical data; retain a hash
    # directly on the diagnostic so its source cannot be confused with test text.
    training = raw.decode('utf-8')
    overlaps = []
    for sample in generation['samples']:
        if sample['greedy'] or (sample['temperature'] == .7 and sample['seed'] == 42):
            overlaps.append({'policy': sample['checkpoint_policy'], 'prompt': sample['prompt'],
                'greedy': sample['greedy'], 'temperature': sample['temperature'], 'seed': sample['seed'],
                'checkpoint_sha256': sample['checkpoint_sha256'],
                **longest_overlap(sample['text'][len(sample['prompt']):], training)})
    assert len(overlaps) == 20
    atomic_json(output / 'fixed20_training_overlaps.json', {
        'selection': 'Both checkpoints, all five prompts, greedy and temperature .7/seed42; no filtering',
        'training_file_sha256': file_hash(root / 'data/processed/tiny-shakespeare-v3/train.txt'),
        'generation_file_sha256': file_hash(output / 'generation.json'),
        'samples': overlaps,
        'limitations': 'Longest literal substring excludes prompt. Common labels/phrases repeat; this is not a corpus-wide memorization rate. Test text was not searched.'})
    f01 = list(csv.DictReader((output / 'validation.csv').open(encoding='utf-8')))
    comparisons = []
    for experiment in [f'E{i:02d}' for i in range(1, 15)]:
        prior = list(csv.DictReader((root / f'report/pilots/{experiment}_seed42/validation.csv').open(encoding='utf-8')))
        assert len(prior) == 20
        for row in prior:
            match = next(r for r in f01 if r['step'] == row['step'])
            comparisons.append({'reference': experiment, 'step': int(row['step']),
                'reference_BPB': float(row['bpb']), 'F01_BPB': float(match['bpb']),
                'absolute_BPB_difference': float(match['bpb']) - float(row['bpb']),
                'relative_BPB_percent': 100 * (float(match['bpb']) / float(row['bpb']) - 1)})
    with (output / 'all_pilots_matched_validation.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader(); writer.writerows(comparisons)
    atomic_json(output / 'all_pilots_matched_validation.json', {'comparisons': comparisons,
        'scope': '280 actual matched observations; architecture/tokenizer/conditioning/cost differ. Not replications.'})
    presentation = output / 'presentation_support'
    atomic_json(presentation / 'parameter_breakdown.json', __import__('scripts.final.protocol', fromlist=['parameter_breakdown']).parameter_breakdown())
    samples = []
    for sample in generation['samples']:
        if sample['checkpoint_policy'] == 'best' and not sample['greedy'] and sample['temperature'] == .7 and sample['seed'] == 42:
            samples.append(sample)
    atomic_json(presentation / 'five_fixed_unedited_examples.json', {
        'selection': 'All five best-checkpoint temperature .7/seed42 outputs, no filtering or editing', 'samples': samples})
    assert len(samples) == 5
    records = read(output / 'master_comparison.json')['runs']
    baseline = next(r for r in records if r['experiment'] == 'E01')
    lines = ['\n## Actual efficiency and baseline-relative prediction\n',
        '![All observed best/final BPB values](comparison_figures/all15_best_final_BPB.png)\n',
        '| Experiment | Best BPB change vs E01 | Parameters vs E01 | Retained training seconds | Updates/s | Peak allocated MiB | Full-validation seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for record in records:
        lines.append(f"| {record['experiment']} | {100*(record['best_BPB']/baseline['best_BPB']-1):+.3f}% | "
            f"{100*(record['parameters']/baseline['parameters']-1):+.3f}% | {record['training_seconds']:.3f} | "
            f"{record['updates_per_training_second']:.4f} | {record['peak_GPU_MiB']:.3f} | {record['validation_seconds']:.3f} |")
    lines.append('''
Costs describe the recorded trajectory, not repeated controlled speed trials.
F01 has 40 validations and 10,000 updates; the pilots have 20 and 5,000.
E07's retained 946.000384 seconds exclude 38.317729 seconds from 207 superseded
attempts: known physical training was at least 984.318113 seconds and 5,207
updates. Later unsaved work is unknown. E12 records no observed replay.
Mean or standard deviation across different architectures would not be a
cross-seed uncertainty estimate; each condition has only one observed seed.

## Factor-level observations and negative outcomes

- **Tokenization:** E02 has the lowest pilot BPB, but BPE changes vocabulary,
  effective byte context, source-byte exposure, parameters and generation length.
  Its token perplexity cannot be ranked against character models. BPE does not
  eliminate dialogue loops, malformed words or invalid sampled byte sequences.
- **Positions:** E03 RoPE improves character-model BPB versus E01; E04 learned
  positions regress while adding parameters. RoPE's greedy repetition remains
  severe in the fixed grid. Lower predictive loss is not a coherence result.
- **Activation and feed-forward capacity:** E05 ReLU and E10 expansion 2× have
  worse recorded best BPB than E01. E10 saves 591,360 parameters but does not
  reproduce baseline prediction quality. These outcomes do not prove GELU or
  larger MLPs always win on other widths, datasets or seeds.
- **Heads:** E06 four heads and E07 eight heads show small BPB reductions versus
  E01 at unchanged width/parameter count. Head dimension changes mechanically;
  the tiny single-seed differences do not justify a general optimum claim.
- **Depth:** E08 six blocks and E09 eight blocks improve best BPB over four-block
  E01 while increasing capacity, memory and time. E09 is stronger predictively
  than E08, but its final score is worse than its selected minimum. Depth cannot
  be isolated from capacity/compute here; repeated or malformed text persists.
- **Context:** E11 context 64 scores better than E01; E12 context 256 scores
  worse. E11 samples 64 windows/update with accumulation 16; E01 samples 32/8;
  E12 samples 16/4. All predict 4,096 targets/update, but conditioning information,
  sequence grouping and compute differ. A shorter context is not automatically
  faster, and these results do not show that long context is intrinsically bad.
- **Width:** E13 width 144 saves 782,976 parameters and worsens prediction;
  E14 width 240 improves best BPB while adding 1,004,160 parameters. Head and MLP
  dimensions and initialization RNG consumption change with width. Memory and
  validation costs differ; neither outcome isolates parameter count alone.

## F01 selection and outcome

The pre-run decision retained BPE, RoPE, eight blocks and width 240 from the
strongest relevant pilot directions, and preserved the explicitly approved
six-head/GELU/MLP4×/context128 settings. It did not sum pilot improvements or
search combinations. [The frozen decision](F01_architecture_decision.md) records
that reasoning before the F01 trajectory began.

F01's early selected BPB is lower than all pilot minima, but its 5,000- and
10,000-update states are worse than the strongest pilots. The extra 5,000 updates
show sustained validation deterioration rather than improved generalization.
Early F01 greedy text remains highly repetitive; later text can look locally
more fluent while showing worse prediction and literal training-passage copying.
Five sampled outputs contain invalid UTF-8 bytes. Complete unedited outputs and
negative observations are preserved in [interpretation.md](interpretation.md).

The single reserved F01 test assessment followed immutable validation selection;
it did not choose between architectures. Repeated validation-based architecture
choices, one contiguous corpus, one seed, different sessions and differing byte
context/exposure restrict generalization. No cross-seed significance, additive
factor attribution, global language understanding or general superiority is
claimed. The former three-seed grid was explicitly discontinued, not completed.
''')
    consolidated = output / 'E01_E14_consolidated_analysis.md'
    prefix = consolidated.read_text(encoding='utf-8').split('\n## Actual efficiency and baseline-relative prediction\n', 1)[0]
    consolidated.write_text((prefix + '\n'.join(lines)).rstrip() + '\n', encoding='utf-8')
    native = output / 'evidence/research'
    completion = read(native / 'completion.json')
    freeze = read(native / 'selection_freeze.json')
    archive = read(native / 'artifact_manifest.json')
    timeline = f'''# F01 training and verification sequence

1. Reviewed main `e4f768d53038662f0f24bcb9cb84a5210ad38a85`; all 14 pilot
   reports consolidated; architecture decision and protocol deviation frozen.
2. Execution source `{completion['source_commit']}` committed and tested.
3. Qualified T4 development notebook `nano-f01-development-20261014-f36bc7a7/1`
   completed; real smoke artifact independently retrieved, CPU verified.
4. Fresh T4 development recovery `nano-f01-development-recovery-20261014-0230bf95/1`
   preserved full state and smoke-only continuation matched the fresh reference.
5. One research notebook `nano-f01-research-20261014-05b5949a/1` completed
   10,000 sequential journal updates and all 40 scheduled validations.
6. Minimum validation BPB selected update {freeze['best_step']}; identity frozen
   before the single reserved test likelihood assessment. All 100 best/final and
   50 internal step-5,000 samples generated by the real model.
7. Saved research output version 1 published archive `{archive['archive_sha256']}`.
8. Fresh T4 recovery `nano-f01-research-recovery-20261014-d22a4fd8/1` restored
   all six complete state fields at 10,000 with zero updates, verified 41 native
   snapshots and exactly reproduced all 100 generation strings/token settings.
9. Independent Windows archive download/CRC/all 82 file hashes, all 40 checkpoint
   counters, sampler replay and best/final CPU inference verified in the report.
10. Figures, full reports and reviewable PR delivered; no additional research run.

This is an ordered evidence timeline, not invented exact timestamps. Training
4,269.206 s; full validation 945.441 s; measured wrapper and archive components
sum 6,317.442 s. Queue, provisioning, publication and download are outside that
sum. The receipts retain the separately measured scopes.
'''
    (presentation / 'training_verification_timeline.md').write_text(timeline, encoding='utf-8')
    (presentation / 'findings_and_limitations.md').write_text(
        (output / 'interpretation.md').read_text(encoding='utf-8'), encoding='utf-8')
    (output / 'acceptance_checklist.md').write_text('''# F01 acceptance evidence

| Gate | Observed evidence |
|---|---|
| Frozen approved architecture | 6,046,560 parameters; BPE1024, width240, eight blocks, six head40, RoPE, GELU, MLP960, context128, dropout.1 |
| Scientific package preserved | Historical Git-blob parity and scientific fingerprint; original E01–E14 configs/reports unchanged |
| CPU and real CUDA numerical audit | Independent rotation/attention/scaling/preLN/CE/causal/gradient/dropout/context checks; actual audit receipts |
| Dataset/tokenizer separation | Canonical hashes/partitions and train-only BPE artifact match; no boundary windows |
| Qualified environment | Python3.11.14, torch2.5.1+cu124, runtimeCUDA12.4, NumPy1.26.4, T4 GPU0, matching freeze |
| Persistent development gate before research | Published development archive/CPU and fresh CUDA full-state plus smoke-only continuity passed |
| Research trajectory | Independent journal and 40 checkpoint counters: exactly10,000 authoritative and known physical research updates, no failure/resume/replay |
| Evaluation | 40 full sliding validations; 55,514 bytes each after256-byte exclusion, context128BPE |
| Sampling and exposure | Independent private-RNG replay, original-byte sums and coverage union match saved state |
| Checkpoints and snapshots | 40 resumable research states; 41 native saved snapshots independently hashed |
| Best selection and test isolation | Update1,750 immutable identity frozen before one reserved test evaluation |
| Generations | Complete100-slot best/final plus50-slot internal5k grids; fresh CUDA exactly reproduces all100strings and token settings |
| Invalid byte outcomes retained | Five invalidUTF8 outputs diagnosed without repair/filtering |
| Durable publication and retrieval | Saved output version1 COMPLETE, independent2,648,712,125-byte archive SHA/CRC/safe paths/exact82-file inventory/allhashes |
| Best/final CPU inference | Genuine independent receipt and checkpoint hashes |
| Fresh CUDA state restoration | All six fields equal, 10,000→10,000, zero additional optimizer updates; best/final CUDA inference |
| Results and comparisons | Actual summary/40-rowCSV, all280 matched pilot observations, all15config master table, genuine figures and detailed interpretation |
| Scope preserved | No further research configurations, seeds, extensions, test-based tuning or automatic PR merge |

Receipts are in evidence/research and evidence/development; the full written
analysis explains exact measurement scopes, failures and scientific limitations.
GitHub review/merge remains a human workflow: a reviewable PR does not mean it
has been reviewed or merged. Delivery CI status must be checked on its actual
commit before claiming success.

**Engineering GO:** this completed single F01 trajectory is recoverable and
documented. **Scientific outcome:** the early predictive target was met, but
generation/coherence limitations and sustained late validation deterioration
remain. This does not authorize another run or claim the former replicated
three-seed study was completed.
''', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); supplement(args.root, args.output)
