"""Bounded, real-checkpoint submission demonstrations using the frozen package."""
import argparse
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import torch
from fastapi.testclient import TestClient
from nano_lm.artifacts import atomic_json
from nano_lm.generate import generate
from api.main import create_app
from api.registry import Registry
from tools.submission_models import checksums

PROMPTS = ['To be, or not to be', 'ROMEO:\n', 'JULIET:\n', 'First Citizen:\n', 'The king']


def environment():
    return {'python': sys.version, 'executable': sys.executable, 'os': platform.platform(),
            'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
            'freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines(),
            'modules': {m: str(Path(importlib.import_module('nano_lm.' + m).__file__).resolve())
                        for m in ['model', 'attention', 'positions', 'bpe', 'generate']}}


def architectures(registry):
    return [{'id': mid, 'selected_step': registry.entries[mid]['checkpoint_step'],
             'parameters': registry.load(mid).parameter_count(), 'architecture': registry.load(mid).config.to_dict(),
             'historical_validation_bpb': registry.entries[mid]['validation_bpb']}
            for mid in registry.entries]


def attention(registry, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model_id, prompt = 'b01', 'ROMEO:\nTo be, or not to be'
    model = registry.load(model_id)
    ids = torch.tensor([registry.tokenizer.encode(prompt)])
    with torch.no_grad():
        logits, _, weights = model(ids, return_attention=True)
        changed = ids.clone()
        boundary = ids.shape[1] // 2
        changed[:, boundary:] = (changed[:, boundary:] + 1) % model.config.vocab_size
        alternate = model(changed)[0]
    error = (logits[:, :boundary] - alternate[:, :boundary]).abs().max().item()
    assert error <= 1e-6
    assert all(torch.count_nonzero(w.triu(1)) == 0 for w in weights)
    values = weights[0][0, 0].numpy()
    labels = [f'{i}: {registry.tokenizer.decode([token])!r}' for i, token in enumerate(ids[0].tolist())]
    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    chart = ax.imshow(values, vmin=0, vmax=1, cmap='viridis')
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
           yticklabels=labels, xlabel='Key token (position and decoded piece)', ylabel='Query token',
           title='Trained B01 step 3000 — layer 0, head 0\n' + repr(prompt))
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    fig.colorbar(chart, ax=ax, label='Causal attention probability')
    path = directory / 'trained_attention.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {'model': model_id, 'checkpoint_step': 3000, 'prompt': prompt, 'input_ids': ids[0].tolist(),
            'input_shape': list(ids.shape), 'attention_shapes': [list(w.shape) for w in weights],
            'head_dimension': model.config.d_model // model.config.n_heads,
            'future_logit_max_abs_error': error, 'future_attention_exactly_zero': True,
            'layer': 0, 'head': 0, 'probabilities': values.tolist(), 'figure': path.name}


def generation(registry):
    rows = []
    for mid in registry.entries:
        model = registry.load(mid)
        reference = json.loads((registry.artifacts / mid / 'metadata.json').read_text(encoding='utf-8'))
        for prompt in PROMPTS:
            for greedy in [True, False]:
                ids = torch.tensor([registry.tokenizer.encode(prompt)])
                start = time.perf_counter()
                output = generate(model, ids, 32, .7, greedy, 42)[0].tolist()
                seconds = time.perf_counter() - start
                token_ids = output[len(ids[0]):]
                canonical = next((c for c in reference['equivalence']['cases']
                                  if c['prompt'] == prompt and c['greedy'] == greedy), None)
                same_ids = canonical is None or token_ids == canonical['token_ids']
                decoded = registry.tokenizer.decode(output)
                if canonical and not same_ids:
                    # Cross-platform sampling can differ despite unchanged weights/math.
                    # Report differences; never silently relabel exact equivalence as passed.
                    status = 'different; platform-sensitive generation, requires review'
                else:
                    status = 'exact reference match' if canonical else 'no prior reference for this prompt'
                rows.append({'model': mid, 'checkpoint_step': registry.entries[mid]['checkpoint_step'],
                             'prompt': prompt, 'prompt_token_ids': ids[0].tolist(), 'token_ids': token_ids,
                             'full_output': decoded, 'greedy': greedy, 'temperature': .7, 'seed': 42,
                             'new_tokens': 32, 'seconds': seconds, 'replacement_character': '\ufffd' in decoded,
                             'reference_status': status, 'reference_exact_ids': same_ids if canonical else None})
    return rows


def api(registry):
    with TestClient(create_app(registry, rate_limit=1000)) as client:
        assert client.get('/ready').status_code == 200
        response = client.post('/v1/generate', json={'model_id': 'b01', 'prompt': 'ROMEO:\n',
                               'max_new_tokens': 32, 'temperature': .7, 'seed': 42, 'greedy': True})
        assert response.status_code == 200
        expected = generate(registry.load('b01'), torch.tensor([registry.tokenizer.encode('ROMEO:\n')]),
                            32, .7, True, 42)[0].tolist()
        assert response.json()['full_output'] == registry.tokenizer.decode(expected)
        return {'status': 200, 'readiness': 200, 'canonical_decoding_equal': True, 'response': response.json()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('model_artifacts'))
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory')
    parser.add_argument('--section', choices=['environment', 'models', 'attention', 'generation', 'api', 'all'], default='all')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    start = time.perf_counter()
    result = {'scope': 'Submission CPU inference; zero research updates or test evaluation',
              'environment': environment()}
    if args.section != 'environment':
        result['files'] = checksums(args.artifacts)
        registry = Registry(args.artifacts)
        result['models'] = architectures(registry)
        prompt = PROMPTS[0]
        ids = registry.tokenizer.encode(prompt)
        assert registry.tokenizer.decode(ids) == prompt
        result['tokenizer'] = {'sha256': registry.tokenizer.checksum, 'vocabulary': registry.tokenizer.vocab_size,
                               'prompt': prompt, 'ids': ids, 'decoded': registry.tokenizer.decode(ids),
                               'context': 128, 'long_prompt_cropped_to_last_tokens': 128}
        if args.section in ['attention', 'all']:
            result['attention'] = attention(registry, args.output)
        if args.section in ['generation', 'all']:
            result['generations'] = generation(registry)
        if args.section in ['api', 'all']:
            result['api'] = api(registry)
    result['measured_seconds'] = time.perf_counter() - start
    atomic_json(args.output / 'receipt.json', result)
    references = [r for r in result.get('generations', []) if r['reference_exact_ids'] is not None]
    print(json.dumps({'receipt': str(args.output / 'receipt.json'), 'models': len(result.get('models', [])),
                      'generation_cases': len(result.get('generations', [])),
                      'reference_cases': len(references),
                      'exact_reference_matches': sum(r['reference_exact_ids'] for r in references),
                      'measured_seconds': result['measured_seconds']}))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
