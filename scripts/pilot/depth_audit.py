"""E08/E09 numerical depth audit; no optimizer updates or scientific code changes."""
import argparse
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from nano_lm.artifacts import atomic_json
from nano_lm.config import load_experiment
from nano_lm.generate import generate
from nano_lm.model import GPT


def audit(root, device='cpu', experiment='E08'):
    assert experiment in {'E08','E09'}
    layers, parameters = (6,2694528) if experiment=='E08' else (8,3584256)
    torch.set_num_threads(1)
    baseline, defaults, _ = load_experiment(root, 'E01', 65)
    config, training, variant = load_experiment(root, experiment, 65)
    assert training == defaults and variant['model_overrides'] == {'n_layers': layers}
    assert {k for k in config.to_dict() if config.to_dict()[k] != baseline.to_dict()[k]} == {'n_layers'}
    torch.manual_seed(42); a = GPT(baseline).to(device)
    torch.manual_seed(42); b = GPT(config).to(device)
    torch.manual_seed(42); repeat = GPT(config).to(device)
    assert a.parameter_count() == 1804800 and b.parameter_count() == parameters
    assert all(torch.equal(v, repeat.state_dict()[k]) for k, v in b.state_dict().items())
    shared = {k: torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()}
    assert len(b.blocks) == layers and config.n_heads == 6 and config.d_model // config.n_heads == 32
    pointers = [p.data_ptr() for block in b.blocks for p in block.parameters()]
    assert len(pointers) == len(set(pointers))
    assert b.position_embedding is None and config.position == 'sinusoidal'
    ids = torch.randint(0, 65, (2, 128), device=device)
    targets = torch.randint(0, 65, (2, 128), device=device)
    b.eval(); x = b.token_embedding(ids) + b.sinusoidal[:128]
    for block in b.blocks:
        assert isinstance(block.norm1, nn.LayerNorm) and isinstance(block.norm2, nn.LayerNorm)
        assert isinstance(block.mlp[1], nn.GELU) and not block.attention.rope
        assert block.dropout.p == block.attention.dropout.p == 0.1
        attended, _ = block.attention(block.norm1(x))
        residual = x + attended
        expected = residual + block.mlp(block.norm2(residual))
        actual, _ = block(x)
        torch.testing.assert_close(actual, expected)
        x = actual
    if experiment=='E09':
        # Independent attention reference checks scaling, Q/K/V shapes and merge.
        for block in b.blocks:
            normalized = block.norm1(x)
            q,k,v = [layer(normalized).reshape(2,128,6,32).transpose(1,2) for layer in (block.attention.q,block.attention.k,block.attention.v)]
            assert all(t.shape==(2,6,128,32) for t in (q,k,v))
            scores = q @ k.transpose(-2,-1) / (32 ** 0.5)
            scores = scores.masked_fill(torch.ones(128,128,device=device,dtype=torch.bool).triu(1),float('-inf'))
            probabilities = scores.softmax(-1)
            merged = (probabilities @ v).transpose(1,2).contiguous().reshape(2,128,192)
            actual_attention, actual_probabilities = block.attention(normalized)
            torch.testing.assert_close(actual_probabilities,probabilities)
            torch.testing.assert_close(actual_attention,block.attention.output(merged))
    with patch('nano_lm.attention.apply_rope', side_effect=AssertionError('RoPE inactive')):
        logits, loss, attentions = b(ids, targets, return_attention=True)
    torch.testing.assert_close(logits, b.head(b.norm(x)))
    torch.testing.assert_close(loss, F.cross_entropy(logits.flatten(0, 1), targets.flatten()))
    assert logits.shape == (2, 128, 65) and torch.isfinite(logits).all() and torch.isfinite(loss)
    assert len(attentions) == layers and all(w.shape == (2, 6, 128, 128) and torch.count_nonzero(w.triu(1)) == 0 for w in attentions)
    changed = ids.clone(); changed[:, 64:] = (changed[:, 64:] + 1) % 65
    torch.testing.assert_close(b(changed)[0][:, :64], logits[:, :64], rtol=1e-5, atol=1e-6)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())
    gradients = []
    for i, block in enumerate(b.blocks):
        assert block.attention.q.weight.grad.norm() > 0
        assert block.attention.v.weight.grad.norm() > 0
        assert block.mlp[0].weight.grad.norm() > 0
        gradients.append({'block': i, 'gradient_norm': float(torch.sqrt(sum(p.grad.square().sum() for p in block.parameters())).item())})
    b.train(); assert not torch.equal(b(ids)[0], b(ids)[0])
    b.eval(); torch.testing.assert_close(b(ids)[0], logits)
    prompt = torch.randint(0, 65, (1, 139), device=device)
    expected = b(prompt[:, -128:])[0][:, -1].argmax(-1)
    torch.testing.assert_close(generate(b, prompt, 1, greedy=True)[:, -1], expected)
    return {'status': 'passed', 'experiment': experiment, 'device': device,
            'parameters': b.parameter_count(), 'additional_parameters': b.parameter_count()-a.parameter_count(),
            'model': config.to_dict(), 'block_count': layers, 'block_gradients': gradients,
            'seed42_shared_E01_parameter_equality': shared,
            'checks': ['sole_depth_override_training_unchanged', f'{layers}_blocks' if experiment=='E09' else 'six_blocks', 'independent_block_storage',
                       'exact_parameter_counts', 'seed42_repeat_determinism', f'all_{layers}_preLN_residual_references' if experiment=='E09' else 'all_six_preLN_residual_references',
                       'GELU_sinusoidal_no_RoPE', f'{layers}_attention_shapes_and_causal_mask' if experiment=='E09' else 'six_attention_shapes_and_causal_mask',
                       'finite_forward_manual_CE', 'future_invariance', 'all_block_finite_nonzero_gradients',
                       'train_eval_dropout', 'cropped_generation'] + (['eight_blocks_QKV_width32_merge192','independent_sqrt32_softmax_value_reference'] if experiment=='E09' else []),
            'limit': 'Development gradients establish connectivity only, not research vanishing/exploding-gradient behavior; extra capacity and initialization order confound depth.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--experiment', choices=['E08','E09'], default='E08'); p.add_argument('--device', default='cpu'); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); atomic_json(a.output, audit(a.root, a.device, a.experiment))
