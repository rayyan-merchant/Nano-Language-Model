"""Independent numerical audit of the locked E03 path; no training updates."""
import argparse
import math
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from nano_lm.artifacts import atomic_json
from nano_lm.attention import apply_rope, split_heads, merge_heads
from nano_lm.config import load_experiment
from nano_lm.generate import generate
from nano_lm.model import GPT


def audit(root, device='cpu'):
    torch.set_num_threads(1)
    baseline, defaults, _ = load_experiment(root, 'E01', 65)
    config, other, variant = load_experiment(root, 'E03', 65)
    assert defaults == other and variant['model_overrides'] == {'position': 'rope'}
    assert {k for k in config.to_dict() if config.to_dict()[k] != baseline.to_dict()[k]} == {'position'}
    assert config.d_model // config.n_heads == 32
    torch.manual_seed(42); a = GPT(baseline).to(device)
    torch.manual_seed(42); b = GPT(config).to(device)
    assert a.parameter_count() == b.parameter_count() == 1804800
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert b.position_embedding is None
    x = torch.randn(2, 6, 11, 32, device=device, requires_grad=True)
    rotated = apply_rope(x); expected = x.detach().clone()
    # Independent scalar 2x2 reference, not implementation helpers.
    for pos in range(11):
        for pair in range(16):
            angle = pos / (10000 ** (2 * pair / 32))
            c, s = math.cos(angle), math.sin(angle)
            even, odd = x.detach()[..., pos, 2*pair], x.detach()[..., pos, 2*pair+1]
            expected[..., pos, 2*pair] = c*even - s*odd
            expected[..., pos, 2*pair+1] = s*even + c*odd
    torch.testing.assert_close(rotated, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(rotated.reshape(2,6,11,16,2).square().sum(-1),
                               x.reshape(2,6,11,16,2).square().sum(-1), rtol=2e-5, atol=2e-6)
    rotated.square().sum().backward()
    torch.testing.assert_close(x.grad, 2*x.detach(), rtol=2e-5, atol=2e-6)
    attention = b.blocks[0].attention; attention.eval()
    inputs = torch.randn(2, 11, 192, device=device); calls = []
    def recording(v):
        calls.append(v.detach().clone()); return apply_rope(v)
    with patch('nano_lm.attention.apply_rope', recording):output, probabilities = attention(inputs)
    assert len(calls) == 2 and all(v.shape == (2,6,11,32) for v in calls)
    q, k, v = [split_heads(layer(inputs), 6) for layer in (attention.q,attention.k,attention.v)]
    torch.testing.assert_close(calls[0], q); torch.testing.assert_close(calls[1], k)
    scores = (apply_rope(q) @ apply_rope(k).transpose(-1,-2)) / math.sqrt(32)
    scores = scores.masked_fill(torch.ones(11,11,device=device,dtype=torch.bool).triu(1), float('-inf'))
    reference = attention.output(merge_heads(scores.softmax(-1) @ v))
    torch.testing.assert_close(output, reference)
    assert probabilities.shape == (2,6,11,11) and torch.count_nonzero(probabilities.triu(1)) == 0
    b.eval(); ids = torch.randint(0,65,(2,128),device=device); targets=torch.roll(ids,-1,1)
    logits, loss, _ = b(ids, targets)
    assert logits.shape == (2,128,65) and torch.isfinite(logits).all() and torch.isfinite(loss)
    torch.testing.assert_close(loss, F.cross_entropy(logits.flatten(0,1),targets.flatten()))
    # Poison the unused buffer to detect accidental additive positions.
    b.sinusoidal.fill_(float('nan'));torch.testing.assert_close(b(ids)[0], logits)
    changed=ids.clone();changed[:,64:] = (changed[:,64:]+1)%65
    torch.testing.assert_close(b(changed)[0][:,:64], logits[:,:64], rtol=1e-5, atol=1e-6)
    loss.backward()
    for block in b.blocks:
        for projection in (block.attention.q,block.attention.k,block.attention.v):
            assert projection.weight.grad is not None and torch.isfinite(projection.weight.grad).all()
            assert projection.weight.grad.norm() > 0
    prompt=torch.randint(0,65,(1,139),device=device)
    wanted=b(prompt[:,-128:])[0][:,-1].argmax(-1)
    torch.testing.assert_close(generate(b,prompt,1,greedy=True)[:,-1],wanted)
    return {'status':'passed','device':device,'parameters':b.parameter_count(),'head_dimension':32,
            'model':config.to_dict(),'same_seed_initial_weights_match_E01':True,
            'checks':['only_position_factor','same_parameter_shapes_count_initial_weights','no_learned_positions',
                      'scalar_adjacent_pair_rotation','pair_norm_preservation','rotation_gradient',
                      'projected_split_QK_only_V_unchanged','causal_mask_shapes','finite_forward_cross_entropy',
                      'sinusoidal_buffer_unused','future_token_invariance','QKV_gradient_flow',
                      'generation_truncation_local_positions'],
            'limit':'Finite local-context correctness; not extrapolation or language-quality evidence'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();atomic_json(a.output,audit(a.root,a.device))
