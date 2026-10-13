"""Describe every fixed-protocol generation; no editing, inference or ranking."""
import argparse
import json
from pathlib import Path
import statistics
import hashlib
import subprocess
from nano_lm.artifacts import atomic_json, file_hash


def describe(root, baseline, candidate, output, experiment="E06"):
    assert experiment in {"E06", "E08"}
    protocol = json.loads((root/'configs/generation_protocol.json').read_text())
    # Windows autocrlf can change working-copy bytes without changing the protocol.
    canonical_bytes = subprocess.check_output(['git', 'show', 'HEAD:configs/generation_protocol.json'], cwd=root)
    assert json.loads(canonical_bytes) == protocol
    protocol_hash = hashlib.sha256(canonical_bytes).hexdigest()
    groups, matched = [], []
    grids = []
    for label, path in [('E01', baseline), (experiment, candidate)]:
        grid = json.loads(path.read_text(encoding='utf8'))
        assert grid['protocol'] == protocol and grid['protocol_sha256'] == protocol_hash
        assert len(grid['samples']) == 100
        entries = {}
        for sample in grid['samples']:
            key = (sample['checkpoint_policy'], sample['prompt'], sample['greedy'], sample['temperature'], sample['seed'])
            assert key not in entries and sample['max_new_tokens'] == 256
            assert sample['text'].startswith(sample['prompt'])
            assert sample['checkpoint_policy'] in {'best', 'final'}
            assert sample['prompt'] in protocol['prompts']
            if sample['greedy']:
                assert sample['temperature'] == 1.0 and sample['seed'] == 42
            else:
                assert sample['temperature'] in protocol['temperatures'] and sample['seed'] in protocol['sampling_seeds']
            text = sample['text'][len(sample['prompt']):]
            assert len(text) == 256  # Both compared models have one ASCII character/token.
            windows = [text[i:i+8] for i in range(max(0, len(text)-7))]
            entries[key] = {'checkpoint': sample['checkpoint'], 'checkpoint_sha256': sample['checkpoint_sha256'],
                            'characters': len(text), 'replacement_characters': text.count('\ufffd'),
                            'repeated_8gram_position_fraction': 1-len(set(windows))/len(windows) if windows else 0}
        grids.append(entries)
        for policy in ['best', 'final']:
            for greedy, temperature in [(True, 1.0)] + [(False, t) for t in protocol['temperatures']]:
                values = [v for k, v in entries.items() if k[0] == policy and k[2:4] == (greedy, temperature)]
                assert len(values) == (5 if greedy else 15)
                groups.append({'experiment': label, 'checkpoint_policy': policy, 'greedy': greedy,
                               'temperature': temperature, 'samples': len(values),
                               'mean_continuation_characters': statistics.mean(v['characters'] for v in values),
                               'mean_repeated_8gram_position_fraction': statistics.mean(v['repeated_8gram_position_fraction'] for v in values),
                               'replacement_characters': sum(v['replacement_characters'] for v in values)})
    assert set(grids[0]) == set(grids[1])
    for key, value in grids[0].items():
        matched.append({'policy': key[0], 'prompt': key[1], 'greedy': key[2], 'temperature': key[3], 'seed': key[4],
                        'E01': value, experiment: grids[1][key]})
    atomic_json(output, {'schema_version': 1, 'protocol_sha256': protocol_hash,
                        'working_copy_protocol_file_sha256': file_hash(root/'configs/generation_protocol.json'),
                        'generation_file_sha256': {'E01': file_hash(baseline), experiment: file_hash(candidate)},
                        'groups': groups, 'matched_settings': matched,
                        'scope': 'All100 rows per genuine checkpoint grid. Repetition is descriptive, not a coherence score, significance result or memorization proof.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    for name in ['baseline', 'candidate', 'output']: p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--experiment',choices=['E06','E08'],default='E06')
    a = p.parse_args(); describe(a.root, a.baseline, a.candidate, a.output,a.experiment)
