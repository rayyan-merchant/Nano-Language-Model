"""Authoritative selected-model identities anchored to committed research evidence."""
import json
from pathlib import Path
from nano_lm.config import load_experiment

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_HASH = '2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55'
LIMITATIONS = 'Single-seed Tiny Shakespeare research. Validation BPB is not a text-quality rating. Outputs can repeat, contain malformed words or invalid UTF-8 bytes.'
SELECTIONS = {
    'e02': ('Compact', 5000, 2173056, 2.02659521, 'report/pilots/E02_seed42/evidence/artifact_manifest.json'),
    'b01': ('Balanced', 3000, 3062784, 1.98493095, 'report/balanced/evidence/research/artifact_manifest.json'),
    'f01': ('Quality', 1750, 6046560, 1.98376817, 'report/final/evidence/research/artifact_manifest.json'),
}
SUMMARIES = {'e02': 'report/pilots/E02_seed42/summary.json',
             'b01': 'report/balanced/summary.json', 'f01': 'report/final/summary.json'}


def catalog():
    entries = {}
    for model_id, (label, step, parameters, bpb, reference) in SELECTIONS.items():
        evidence = json.loads((ROOT / reference).read_text(encoding='utf-8'))
        summary = json.loads((ROOT / SUMMARIES[model_id]).read_text(encoding='utf-8'))
        if (summary['parameters'] != parameters or summary['best_validation']['step'] != step
                or round(summary['best_validation']['bpb'], 8) != bpb
                or summary['source_commit'] != evidence['source_commit']):
            raise ValueError('Accepted research summary and selected identity disagree')
        checkpoint = next(p for p in evidence['files'] if p.endswith(f'/checkpoints/step_{step:06d}.pt'))
        config, _, _ = load_experiment(ROOT, model_id.upper(), 1024)
        entries[model_id] = {
            'id': model_id, 'name': f'NanoLM {label}', 'description': {
                'e02': 'The original four-block Byte-BPE reference.',
                'b01': 'Six blocks with rotary positions; the default writing model.',
                'f01': 'A wider eight-block model selected before prolonged-training deterioration.',
            }[model_id],
            'parameters': parameters, 'checkpoint_step': step, 'validation_bpb': bpb,
            'architecture': config.to_dict(), 'tokenizer_hash': TOKENIZER_HASH,
            'tokenizer': 'Canonical train-only Byte-BPE / 1024', 'context_length': 128,
            'original_checkpoint_sha256': evidence['files'][checkpoint],
            'training_commit': evidence['source_commit'], 'checkpoint_identity': checkpoint,
            'research_reference': reference, 'limitations': LIMITATIONS,
            'research_summary_reference': SUMMARIES[model_id],
        }
    return entries
