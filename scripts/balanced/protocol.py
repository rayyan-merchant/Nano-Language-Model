"""Narrow B01 authorization; no scientific implementation changes."""
import json
import subprocess
from pathlib import Path
from nano_lm.artifacts import digest
from nano_lm.config import load_experiment
from scripts.final.protocol import provenance as original_provenance, BPE

BASE = 'd43cfbf7e3af6c982a8bcf1509a5f45b8e17ef4b'
CORE = 'd5b205aecec5913cd11d921c0a84899badd95a014a490a7beaab3d2598c940f8'
MODEL = dict(vocab_size=1024, d_model=192, n_heads=6, n_layers=6, context_length=128,
             mlp_expansion=4, activation='gelu', position='rope', dropout=.1)
PARAMETERS = 3062784
MAXIMUM = 5000
STOP_POLICY = dict(metric='full_sliding_validation_BPB', interval=250, minimum_updates=2000,
                   patience=6, minimum_improvement=.001,
                   comparison='bpb < patience_reference - minimum_improvement',
                   best_selection='absolute_minimum_without_tolerance')


def configuration(root):
    config, defaults, variant = load_experiment(Path(root), 'B01', 1024)
    assert config.to_dict() == MODEL and variant['tokenizer'] == 'bpe'
    assert variant['authorized_seed'] == 42 and variant['authorized_maximum_optimizer_updates'] == MAXIMUM
    assert variant['early_stopping'] == STOP_POLICY
    assert defaults['optimizer'] == 'AdamW' and defaults['scheduler'] == 'constant'
    return config


def provenance(root):
    root = Path(root); result = original_provenance(root)
    assert result['scientific_core_fingerprint'] == CORE
    for name in ['experiments/configs/F01.json']:
        old = subprocess.check_output(['git', 'show', f'{BASE}:{name}'], cwd=root).decode().replace('\r\n', '\n')
        assert (root / name).read_text(encoding='utf-8') == old
    changed = subprocess.check_output(['git', 'diff', '--name-only', BASE, '--',
        'report/pilots', 'report/final', 'src', 'configs'], cwd=root, text=True).splitlines()
    if (root/'docs/maintenance/public_tree.csv').exists():
        from scripts.maintenance.archive_refs import verify_retained_history
        verify_retained_history(root, 'report/pilots/')
        verify_retained_history(root, 'report/final/')
    else:
        assert not changed, f'Historical evidence or scientific source changed: {changed}'
    return {**result, 'reviewed_pre_B01_commit': BASE, 'unchanged_F01_and_pilot_evidence': True,
            'stopping_policy_hash': digest(STOP_POLICY)}


def parameter_breakdown():
    d, h = 192, 768
    block = dict(qkv_projections=3*(d*d+d), attention_output=d*d+d,
                 two_pre_LayerNorms=4*d, MLP_first=d*h+h, MLP_second=h*d+d)
    total = 2*1024*d + 6*sum(block.values()) + 2*d
    assert total == PARAMETERS
    return dict(token_embeddings=1024*d, per_block=block, six_blocks=6*sum(block.values()),
                final_LayerNorm=2*d, untied_LM_head=1024*d, total=total)
