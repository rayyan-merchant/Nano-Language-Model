"""Numerical E06 four-head audit; no optimizer updates or core modifications."""
import argparse
from pathlib import Path
from unittest.mock import patch
import math
import torch
from torch.nn import functional as F
from nano_lm.artifacts import atomic_json
from nano_lm.attention import split_heads, merge_heads, scaled_attention
from nano_lm.config import load_experiment
from nano_lm.generate import generate
from nano_lm.model import GPT


def audit(root, device='cpu'):
    torch.set_num_threads(1)
    baseline, defaults, _ = load_experiment(root, 'E01', 65)
    config, training, variant = load_experiment(root, 'E06', 65)
    assert training == defaults and variant['model_overrides'] == {'n_heads': 4}
    assert {k for k in config.to_dict() if config.to_dict()[k] != baseline.to_dict()[k]} == {'n_heads'}
    assert config.n_heads == 4 and config.d_model // config.n_heads == 48
    torch.manual_seed(42); a = GPT(baseline).to(device)
    torch.manual_seed(42); b = GPT(config).to(device)
    assert a.parameter_count() == b.parameter_count() == 1804800
    assert all(torch.equal(a.state_dict()[k], v) for k, v in b.state_dict().items())
    assert b.position_embedding is None and config.position == 'sinusoidal'
    assert all(not block.attention.rope for block in b.blocks)
    x = torch.randn(2, 128, 192, device=device)
    q, k, v = [split_heads(layer(x), 4) for layer in (b.blocks[0].attention.q, b.blocks[0].attention.k, b.blocks[0].attention.v)]
    assert q.shape == k.shape == v.shape == (2, 4, 128, 48)
    torch.testing.assert_close(merge_heads(split_heads(x, 4)), x)
    scores = q @ k.transpose(-2, -1) / math.sqrt(48)
    mask = torch.ones(128, 128, device=device, dtype=torch.bool).triu(1)
    expected = scores.masked_fill(mask, float('-inf')).softmax(-1)
    attended, weights = scaled_attention(q, k, v)
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(attended, expected @ v)
    for wrong in (32, 192):
        wrong_weights = (q @ k.transpose(-2, -1) / math.sqrt(wrong)).masked_fill(mask, float('-inf')).softmax(-1)
        assert not torch.allclose(weights, wrong_weights)
    b.eval(); ids = torch.randint(0, 65, (2, 128), device=device)
    targets = torch.randint(0, 65, (2, 128), device=device)
    observed = []
    handle = b.dropout.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone()))
    with patch('nano_lm.attention.apply_rope', side_effect=AssertionError('RoPE must be inactive')):
        logits, loss, attentions = b(ids, targets, return_attention=True)
    handle.remove()
    torch.testing.assert_close(observed[0], b.token_embedding(ids) + b.sinusoidal[:128])
    assert logits.shape == (2, 128, 65) and torch.isfinite(logits).all() and torch.isfinite(loss)
    assert len(attentions) == 4 and all(w.shape == (2, 4, 128, 128) and torch.count_nonzero(w.triu(1)) == 0 for w in attentions)
    torch.testing.assert_close(loss, F.cross_entropy(logits.flatten(0, 1), targets.flatten()))
    changed = ids.clone(); changed[:, 64:] = (changed[:, 64:] + 1) % 65
    torch.testing.assert_close(b(changed)[0][:, :64], logits[:, :64], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(b(ids)[0], logits)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())
    assert all(layer.weight.grad.norm() > 0 for block in b.blocks for layer in (block.attention.q, block.attention.k, block.attention.v))
    b.train(); assert not torch.equal(b(ids)[0], b(ids)[0])
    assert all(block.attention.dropout.p == 0.1 for block in b.blocks)
    b.eval(); prompt = torch.randint(0, 65, (1, 139), device=device)
    next_id = b(prompt[:, -128:])[0][:, -1].argmax(-1)
    torch.testing.assert_close(generate(b, prompt, 1, greedy=True)[:, -1], next_id)
    return {'status': 'passed', 'experiment': 'E06', 'device': device, 'parameters': 1804800,
            'model': config.to_dict(), 'head_shape': [2, 4, 128, 48], 'scale_denominator': math.sqrt(48),
            'all_initial_parameter_values_match_E01': True,
            'checks': ['sole_head_count_override', 'parameters_and_initial_values_equal', 'QKV_head_shapes',
                       'split_merge_identity', 'manual_sqrt48_attention', 'wrong_scaling_rejected',
                       'sinusoidal_only_no_RoPE', 'causal_mask', 'finite_forward_manual_CE',
                       'future_token_invariance', 'finite_QKV_backward', 'train_eval_dropout', 'cropped_generation'],
            'limit': 'Numerical fixed-context correctness; not language quality or statistical evidence'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); atomic_json(args.output, audit(args.root, args.device))
