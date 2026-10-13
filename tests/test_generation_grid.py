"""Reporting regression for platform newline conversion and incomplete grids."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from scripts.pilot.compare_generation_grid import describe


@pytest.mark.parametrize('experiment',['E06','E10','E11','E13','E14'])
def test_grid_accepts_windows_newlines_and_rejects_missing_samples(tmp_path,experiment):
    protocol = json.loads((Path(__file__).resolve().parents[1]/'configs/generation_protocol.json').read_text())
    canonical = (json.dumps(protocol, indent=2)+'\n').encode()
    configs = tmp_path/'configs'; configs.mkdir()
    (configs/'generation_protocol.json').write_bytes(canonical.replace(b'\n', b'\r\n'))
    samples = []
    settings = [(True, 1.0, 42)] + [(False, t, s) for t in protocol['temperatures'] for s in protocol['sampling_seeds']]
    for policy in ['best', 'final']:
        for prompt in protocol['prompts']:
            for greedy, temperature, seed in settings:
                samples.append({'checkpoint_policy': policy, 'checkpoint': 'fixture.pt', 'checkpoint_sha256': 'a'*64,
                                'prompt': prompt, 'text': prompt+'a'*256, 'max_new_tokens': 256,
                                'greedy': greedy, 'temperature': temperature, 'seed': seed})
    fixture = {'protocol': protocol, 'protocol_sha256': hashlib.sha256(canonical).hexdigest(), 'samples': samples}
    grid = tmp_path/'synthetic_grid.json'; grid.write_text(json.dumps(fixture))
    with patch('scripts.pilot.compare_generation_grid.subprocess.check_output', return_value=canonical):
        describe(tmp_path, grid, grid, tmp_path/'fixture_summary.json',experiment=experiment)
        summary = json.loads((tmp_path/'fixture_summary.json').read_text())
        assert summary['protocol_sha256'] != summary['working_copy_protocol_file_sha256']
        assert len(summary['matched_settings']) == 100
        fixture['samples'] = samples[:-1]; grid.write_text(json.dumps(fixture))
        with pytest.raises(AssertionError):
            describe(tmp_path, grid, grid, tmp_path/'incomplete.json',experiment=experiment)
