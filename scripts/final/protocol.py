"""Narrow F01 authorization and byte-verifiable historical scientific provenance."""
import json
import subprocess
from pathlib import Path
from nano_lm.artifacts import digest, file_hash
from nano_lm.config import load_experiment

BASE = 'e4f768d53038662f0f24bcb9cb84a5210ad38a85'
AUDITED = 'cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6'
BPE = '2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55'
MODEL = dict(vocab_size=1024, d_model=240, n_heads=6, n_layers=8,
             context_length=128, mlp_expansion=4, activation='gelu', position='rope', dropout=0.1)
PARAMETERS = 6046560
TARGET = 10000
VALIDATION_STEPS = list(range(250, TARGET + 1, 250))


def provenance(root):
    root = Path(root)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE], cwd=root, text=True).splitlines()
    original = {}; core = {}; hashes = {}
    for name in names:
        if (name.startswith('src/') and name.endswith('.py')) or (
                name.startswith(('configs/', 'experiments/configs/')) and name.endswith('.json')):
            raw = subprocess.check_output(['git', 'show', f'{BASE}:{name}'], cwd=root)
            historical = raw.decode('utf-8').replace('\r\n', '\n')
            current = (root / name).read_text(encoding='utf-8')
            if current != historical:
                raise ValueError(f'Frozen scientific source/configuration changed: {name}')
            original[name] = historical
            if name.startswith('src/'):
                core[name] = historical
                hashes[name] = file_hash(root / name)
    if digest(original) != AUDITED:
        raise ValueError('Historical aggregate fingerprint differs from approved reference')
    actual_source = sorted(p.relative_to(root).as_posix() for p in (root/'src').rglob('*.py'))
    if actual_source != sorted(core):
        raise ValueError('Unexpected scientific implementation file')
    return dict(frozen_pre_F01_commit=BASE, audited_pre_F01_aggregate_hash=AUDITED,
                scientific_core_fingerprint=digest(core), scientific_core_files=hashes,
                definition='SHA256 of canonical JSON mapping relative source paths to UTF8 text with universal newlines',
                unchanged_historical_scientific_files=True)


def configuration(root):
    config, defaults, variant = load_experiment(Path(root), 'F01', 1024)
    if config.to_dict() != MODEL or variant['tokenizer'] != 'bpe':
        raise ValueError('F01 configuration is not the authorized architecture')
    if variant['authorized_seed'] != 42 or variant['authorized_optimizer_updates'] != TARGET:
        raise ValueError('F01 authorization mismatch')
    if defaults['optimizer'] != 'AdamW' or defaults['scheduler'] != 'constant':
        raise ValueError('Optimizer/schedule changed')
    return config


def parameter_breakdown():
    width, hidden = 240, 960
    block = {'qkv_projections': 3*(width*width+width),
             'attention_output': width*width+width, 'two_pre_LayerNorms': 4*width,
             'MLP_first': width*hidden+hidden, 'MLP_second': hidden*width+width}
    total = 2*1024*width+8*sum(block.values())+2*width
    assert total == PARAMETERS
    return {'token_embeddings': 1024*width, 'per_block': block,
            'eight_blocks': 8*sum(block.values()), 'final_LayerNorm': 2*width,
            'untied_LM_head': 1024*width, 'total': total}
