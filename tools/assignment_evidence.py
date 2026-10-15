"""Rebuild submission diagrams and audit existing receipts; no training or downloads.

Run from a checkout: python -m tools.assignment_evidence
Educational tensors are explicitly separated from trained-checkpoint evidence.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import textwrap

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from nano_lm.attention import scaled_attention, split_heads, merge_heads
from nano_lm.config import ModelConfig
from nano_lm.model import GPT
from nano_lm.positions import sinusoidal, apply_rope
from api.schemas import GenerateRequest, GenerationResult

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/submission/evidence'
RECEIPTS = ROOT / 'docs/submission/verification/colab-final/receipts'


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def sha(path):
    data = path.read_bytes()
    # Git's existing checkout normalization differs on Windows and Linux. This
    # applies only to audit source text, never model/tokenizer package checksums.
    if path.suffix in {'.json', '.csv', '.txt'}:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def manual_attention():
    """Independent scalar-loop reference versus the frozen tensor implementation."""
    vectors = [[1., 0.], [0., 1.], [1., 1.]]
    scores = [[sum(x*y for x, y in zip(q, k))/math.sqrt(2)
               for k in vectors] for q in vectors]
    result = {'Q': vectors, 'K': vectors, 'V': vectors, 'scaled_scores': scores}
    for causal in [False, True]:
        probabilities, outputs = [], []
        for i, row in enumerate(scores):
            visible = row[:i+1] if causal else row
            exps = [math.exp(x-max(visible)) for x in visible]
            p = [x/sum(exps) for x in exps] + [0.]*(3-len(visible))
            probabilities.append(p)
            outputs.append([sum(p[j]*vectors[j][d] for j in range(3)) for d in range(2)])
        tensor = torch.tensor(vectors, dtype=torch.float64)
        actual, weights = scaled_attention(tensor, tensor, tensor, causal=causal)
        np.testing.assert_allclose(weights.numpy(), probabilities, atol=1e-14)
        np.testing.assert_allclose(actual.numpy(), outputs, atol=1e-14)
        result['causal' if causal else 'bidirectional'] = {'weights': probabilities, 'output': outputs}
    return result


def save(fig, name):
    fig.savefig(OUT / name, dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def page(name, title, text, size=(11, 8)):
    fig, ax = plt.subplots(figsize=size)
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=17, pad=20)
    ax.text(0, 1, text, va='top', family='monospace', fontsize=10.5, linespacing=1.55,
            transform=ax.transAxes)
    save(fig, name)


def matrix(ax, values, title, labels=None):
    values = np.array(values)
    ax.imshow(values, cmap='Blues', vmin=0, vmax=1)
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f'{value:.3f}' if values.dtype != bool else ('allow' if value else 'block'),
                ha='center', va='center', color='white' if value > .6 else 'black')
    ticks = labels or ['1', '2', '3']
    ax.set(xticks=range(len(ticks)), yticks=range(len(ticks)), xticklabels=ticks,
           yticklabels=ticks, xlabel='Key / information source', ylabel='Query / receiving position', title=title)


def trace_models(manifest):
    traces = {}
    with torch.random.fork_rng():
        torch.manual_seed(42)
        for mid, record in manifest['models'].items():
            model = GPT(ModelConfig(**record['architecture'])).eval()
            assert model.parameter_count() == record['parameters']
            shapes, hooks, captured = {}, [], {}
            modules = {'embedding': model.token_embedding, 'Q': model.blocks[0].attention.q,
                       'K': model.blocks[0].attention.k, 'V': model.blocks[0].attention.v,
                       'output_projection': model.blocks[0].attention.output, 'logits': model.head}
            def capture(name):
                def hook(module, args, output):
                    shapes[name] = list(output.shape)
                    captured[name] = output.detach()
                return hook
            for name, module in modules.items():
                hooks.append(module.register_forward_hook(capture(name)))
            ids = torch.arange(12).reshape(1, 12)
            with torch.no_grad():
                logits, _, attention = model(ids, return_attention=True)
                q, k, v = [split_heads(captured[name], model.config.n_heads) for name in ['Q','K','V']]
                if model.config.position == 'rope':
                    q, k = apply_rope(q), apply_rope(k)
                scores = (q @ k.transpose(-2, -1))/math.sqrt(q.shape[-1])
                attended, weights = scaled_attention(q, k, v)
                torch.testing.assert_close(weights, attention[0])
                original_output = captured['output_projection'].clone()
                torch.testing.assert_close(model.blocks[0].attention.output(merge_heads(attended)), original_output)
                shapes['head_split'] = list(q.shape)
                shapes['scaled_scores'] = list(scores.shape)
                shapes['attention_weights'] = list(weights.shape)
                shapes['per_head_output'] = list(attended.shape)
                shapes['concatenated'] = list(merge_heads(attended).shape)
                assert torch.count_nonzero(attention[0].triu(1)) == 0
                assert torch.isfinite(logits).all()
            for hook in hooks:
                hook.remove()
            traces[mid] = {'parameters': model.parameter_count(), 'shapes': shapes,
                           'head_dimension': model.config.d_model // model.config.n_heads,
                           'scope': 'Fresh CPU architecture trace, not trained attention or generation'}
    return traces


def audit_receipts():
    attention = read('docs/submission/verification/colab-final/receipts/attention/receipt.json')
    a = attention['attention']
    assert a['model'] == 'b01' and a['checkpoint_step'] == 3000
    p = np.array(a['probabilities'])
    assert p.shape == (10, 10) and not np.count_nonzero(np.triu(p, 1))
    np.testing.assert_allclose(p.sum(1), 1, atol=2e-7)
    assert a['future_logit_max_abs_error'] == 0
    generations = read('docs/submission/verification/colab-final/receipts/generation/receipt.json')['generations']
    assert len(generations) == 30
    api = read('docs/submission/verification/colab-final/receipts/api/receipt.json')['api']
    assert api['status'] == 200 and api['canonical_decoding_equal']
    response = GenerationResult(**api['response'])
    request = GenerateRequest(model_id='b01', prompt='ROMEO:\n', max_new_tokens=32,
                              temperature=.7, greedy=True, seed=42)
    expected = next(g for g in generations if g['model'] == 'b01' and g['greedy']
                    and g['prompt'] == request.prompt)
    assert expected['full_output'] == response.full_output
    notebooks = []
    for path in sorted((ROOT / 'docs/submission/verification').glob('colab-*/NanoLM*executed.ipynb')):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        assert len(code) == 13 and all(c.get('execution_count') is not None for c in code)
        assert not any(o.get('output_type') == 'error' for c in code for o in c.get('outputs', []))
        notebooks.append({'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha(path),
                          'executed_code_cells': len(code)})
    # Only the final executed notebook is retained; the earlier run has receipts.
    assert len(notebooks) == 1
    log = (ROOT / 'docs/submission/verification/colab-v3/full-tests.txt').read_text(encoding='utf-8')
    assert '151 passed, 1 skipped' in log
    manifest = read('docs/submission/models.json')
    executions = []
    for base in ['colab-v3', 'colab-final/receipts']:
        prefix = 'docs/submission/verification/' + base
        downloaded = read(prefix + '/download_receipt.json')
        assert downloaded['revision'] == manifest['revision'] and len(downloaded['files']) == 6
        for file in downloaded['files']:
            mid, name = file['path'].split('/')
            key = 'artifact' if name == 'model.safetensors' else 'metadata'
            assert file['sha256'] == manifest['models'][mid][key + '_sha256']
            assert file['bytes'] == manifest['models'][mid][key + '_bytes']
        cases = read(prefix + '/generation/receipt.json')['generations']
        assert len(cases) == 30 and sum(c['reference_exact_ids'] is True for c in cases) == 12
        for section in ['models', 'restart']:
            models = read(prefix + '/' + section + '/receipt.json')['models']
            assert len(models) == 3
            for model in models:
                reference = manifest['models'][model['id']]
                assert model['parameters'] == reference['parameters']
                assert model['selected_step'] == reference['checkpoint_step']
                assert model['architecture'] == reference['architecture']
        executions.append({'receipt_directory': prefix, 'files': 6, 'generation_cases': 30,
                           'exact_available_reference_matches': 12, 'fresh_process_models': 3})
    transfer = read('docs/submission/verification/colab-v3-transfer.json')
    assert transfer['executed_code_cells'] == 13 and transfer['cell_errors'] == 0
    with (ROOT / 'report/final/validation.csv').open(newline='', encoding='utf-8') as file:
        validations = list(csv.DictReader(file))
    assert [int(v['step']) for v in validations] == list(range(250,10001,250))
    best = min(validations, key=lambda v: float(v['loss']))
    final_summary = read('report/final/summary.json')
    assert int(best['step']) == final_summary['best_validation']['step'] == 1750
    np.testing.assert_allclose(float(best['bpb']), final_summary['best_validation']['bpb'], atol=1e-12)
    return {'trained_attention_rows_normalized': True, 'future_attention_exactly_zero': True,
            'API_response_matches_saved_generation': True, 'generation_cases': len(generations),
            'notebooks': notebooks, 'independent_executions': executions,
            'F01_validation_observations': len(validations),
            'Linux_test_log': '151 passed, 1 skipped (historical executed receipt)'}, generations, request, response


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    math_result = manual_attention()
    manifest = read('docs/submission/models.json')
    traces = trace_models(manifest)
    receipt_audit, generations, request, response = audit_receipts()
    fmt = lambda x: np.array2string(np.array(x), precision=6, suppress_small=True)
    text = '\n\n'.join([f'Q = K = V (3 x 2):\n{fmt(math_result["Q"])}',
        'QK^T = [[1,0,1], [0,1,1], [1,1,2]]',
        'S = QK^T / sqrt(2):\n' + fmt(math_result['scaled_scores']),
        'Causal S: [[0.707107,-inf,-inf], [0,0.707107,-inf], [0.707107,0.707107,1.414214]]',
        'Row softmax, causal:\n' + fmt(math_result['causal']['weights']),
        'Causal weights @ V (3 x 2):\n' + fmt(math_result['causal']['output']),
        'Token 1 sees 1; token 2 sees 1,2; token 3 sees 1,2,3.\nTyped companion, not a replacement handwritten submission.'])
    page('figure_01_attention_derivation.png', '1 | Three-token self-attention — corrected companion', text, (11, 11))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
    matrix(axes[0], np.tril(np.ones((3, 3), dtype=bool)), 'Allowed: diagonal and past')
    matrix(axes[1], math_result['causal']['weights'], 'After masked softmax')
    fig.suptitle('2 | Implementation uses Boolean upper triangle → scores = −∞ → probability = 0')
    save(fig, 'figure_02_causal_mask.png')
    shape_text = 'Actual E02 configuration; B=1, T=12, D=192, H=6, head=32\nStructural CPU trace with fresh weights, not a trained attention claim.\n\n'
    shape_text += '\n'.join(f'{key:24} {value}' for key, value in traces['e02']['shapes'].items())
    shape_text += '\n\nEach head: softmax(Q K^T / sqrt(32) + causal mask) V.\nIndependent learned Q/K/V subspaces; merge returns width 192.\nSinusoidal addition preserves embedding shape (1,12,192).'
    page('figure_03_tensor_shapes.png', '3 | Actual multi-head tensor shapes', shape_text)
    encoding = sinusoidal(128, 192).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    im = axes[0].imshow(encoding, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set(xlabel='Embedding dimension', ylabel='Position', title='Actual sinusoidal(128,192)')
    fig.colorbar(im, ax=axes[0])
    for d in [0, 1, 16, 17, 64, 65]:
        axes[1].plot(encoding[:, d], label=f'd={d}')
    axes[1].set(xlabel='Position', ylabel='Value', title='Higher dimension pairs vary more slowly')
    axes[1].legend(ncol=2)
    fig.suptitle('4 | PE(p,2i)=sin(p / 10000^(2i/D)); odd dimensions use cosine')
    save(fig, 'figure_04_sinusoidal.png')
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.axis('off')
    ax.set_title('5 | Actual decoder architecture and constructed parameter counts', fontsize=16, pad=22)
    stages = [(.92, 'Text → byte-BPE IDs\nVocabulary 1024; train-only fitting'),
              (.77, 'Token embedding\nIDs (B,T) → representations (B,T,D)'),
              (.62, 'Position information\nE02: additive sinusoidal\nB01/F01: RoPE inside attention, projected Q/K only'),
              (.42, 'Repeat pre-LN causal block L times\nu = x + Dropout(Attention(LN(x)))\nz = u + Dropout(Linear₂(GELU(Linear₁(LN(u)))))'),
              (.22, 'Final LayerNorm → untied vocabulary projection'),
              (.08, 'Logits (B,T,1024)\nShifted-target CE in training; next-token decoding in inference')]
    for i, (y, label) in enumerate(stages):
        ax.text(.29, y, label, ha='center', va='center', fontsize=10,
                bbox={'boxstyle': 'round,pad=.65', 'facecolor': '#edf3f7', 'edgecolor': '#20546e'}, transform=ax.transAxes)
        if i:
            previous_y = stages[i-1][0]
            ax.annotate('', xy=(.29, y+.055), xytext=(.29, previous_y-.055),
                        arrowprops={'arrowstyle':'->','color':'#20546e'}, xycoords='axes fraction')
    lines = []
    for mid, m in manifest['models'].items():
        c = m['architecture']
        lines += [f'{mid.upper()} — selected step {m["checkpoint_step"]}', f'{m["parameters"]:,} parameters',
                  f'D={c["d_model"]}; L={c["n_layers"]}; H={c["n_heads"]}',
                  f'MLP hidden={c["d_model"]*4}; context=128', f'{c["position"]}; GELU; dropout=.1', '']
    ax.text(.67, .96, '\n'.join(lines), va='top', fontsize=11, linespacing=1.6, transform=ax.transAxes)
    save(fig, 'figure_05_architecture.png')
    selected = [g for g in generations if g['model'] == 'b01' and g['greedy']]
    assert len(selected) == 5
    sample_text = '# Figure 8 — Predetermined genuine examples\n\nAll five B01 greedy prompts in receipt order, without filtering or rewriting. '
    sample_text += 'Selected step 3000, 32 new BPE tokens, seed 42, temperature 0.7 (ignored by greedy), context 128. '
    sample_text += 'Final submission CPU demonstration, not the original 256-token research grid.\n\n'
    panels = []
    for g in selected:
        sample_text += f'## Prompt `{g["prompt"]!r}`\n\n```text\n{g["full_output"]}\n```\n\n'
        panels.append('Prompt: ' + repr(g['prompt']) + '\n' + '\n'.join(textwrap.wrap(g['full_output'], 95, replace_whitespace=False)))
    sample_text += 'Source: [unaltered generation receipt](../verification/colab-final/receipts/generation/receipt.json). '
    sample_text += 'Repeated phrases and incomplete endings are preserved; 32-token truncation limits coherence judgments.\n'
    (OUT / 'figure_08_generations.md').write_text(sample_text, encoding='utf-8')
    page('figure_08_generations.png', '8 | B01 step 3000 — all five predetermined greedy examples',
         '32 new tokens; seed 42; context 128; unedited saved outputs\n\n' + '\n\n'.join(panels), (12, 11))
    with (ROOT / 'report/final/master_comparison.csv').open(newline='', encoding='utf-8') as file:
        rows = {r['experiment']: r for r in csv.DictReader(file)}
    for i in range(1,15):
        eid = f'E{i:02}'
        summary = read(f'report/pilots/{eid}_seed42/summary.json')
        assert summary['parameters'] == int(rows[eid]['parameters'])
        assert summary['completed_optimizer_updates'] == int(rows[eid]['updates']) == 5000
        np.testing.assert_allclose(summary['best_validation']['bpb'], float(rows[eid]['best_BPB']), atol=1e-10)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')
    for ax, (title, ids, key) in zip(axes, [('Depth', ['E01','E08','E09'], 'n_layers'),
                                          ('Heads', ['E06','E01','E07'], 'n_heads'),
                                          ('Context', ['E11','E01','E12'], 'context_length')]):
        values = [float(rows[e]['best_BPB']) for e in ids]
        ax.bar([rows[e][key] for e in ids], values, color=['#7293ae', '#20546e', '#58a096'])
        for i, (e, v) in enumerate(zip(ids, values)):
            ax.text(i, v+.007, f'{e}\n{v:.4f}', ha='center', fontsize=9)
        ax.set(title=title, xlabel=key, ylabel='Best validation conditional BPB', ylim=(0, 2.6))
    fig.suptitle('9 | Controlled comparisons, seed 42 only — lower BPB is better\nCapacity, conditioning and compute caveats accompany the evidence index')
    save(fig, 'figure_09_ablations.png')
    fields = list(next(iter(rows.values())).keys())
    with (OUT / 'figure_09_ablations.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows[f'E{i:02}'] for i in range(1,15))
    api_text = 'Existing executed TestClient request: POST /v1/generate\n' + json.dumps(request.model_dump(), indent=2)
    api_text += '\n\nHTTP 200; B01 selected checkpoint 3000; 3,062,784 parameters\nContext capacity 128; actual prompt context 3 tokens.\n\nGenuine response.full_output:\n' + response.full_output
    api_text += '\n\nReceipt request_id: ' + response.request_id + '\nActual response, not a mock; this receipt is local API evidence, not live hosting uptime.'
    page('figure_10_api.png', '10 | Genuine trained-model API inference', api_text, (11, 9))
    fig, axes = plt.subplots(1,2,figsize=(11,4.7), layout='constrained')
    matrix(axes[0], np.ones((3,3), dtype=bool), 'MLM encoder: all positions visible', ['I','[MASK]','tea'])
    matrix(axes[1], np.tril(np.ones((3,3), dtype=bool)), 'CLM decoder: future excluded', ['I','like','tea'])
    fig.suptitle('MLM vs CLM | Exact educational visibility example — no trained BERT used')
    save(fig, 'mlm_vs_clm.png')
    inputs = ['docs/submission/models.json', 'docs/handwritten-assignment.pdf',
              'report/final/master_comparison.csv', 'report/final/summary.json',
              'report/final/validation.csv', 'report/final/figures/combined_loss.png']
    inputs += [str(p.relative_to(ROOT)).replace('\\','/') for p in RECEIPTS.rglob('*') if p.is_file()]
    inputs += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT / 'docs/submission/verification/colab-v3').rglob('*') if p.is_file()]
    inputs += [f'report/pilots/E{i:02}_seed42/summary.json' for i in range(1,15)]
    receipt = {'scope': 'New educational diagrams and independent inspection of historical executed receipts; zero training',
               'starting_main': 'ee2b941311cfb451a73dbc65207ad3dfabb5977f',
               'manual_attention': math_result, 'architecture_traces': traces, 'historical_receipt_checks': receipt_audit,
               'audit_source_hash_policy': 'JSON/CSV/TXT source hashes normalize CRLF to LF to match Git across Windows/Linux; binary PDF/PNG hashes are exact. Model/package verification uses original exact-byte hashes, never this normalization.',
               'inputs_sha256': {p: sha(ROOT / p) for p in sorted(inputs)},
               'inputs_observed_Windows_bytes_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(inputs)},
               'generated_sha256': {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.suffix in {'.png','.csv'}},
               'public_API_request': request.model_dump()}
    (OUT.parent / 'verification/assignment_audit.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'generated_files': len(receipt['generated_sha256']), 'receipt_checks': receipt_audit}, indent=2))


if __name__ == '__main__':
    build()
