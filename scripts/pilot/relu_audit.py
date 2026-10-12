"""Independent canonical E05 audit; no optimizer updates or model changes."""
import argparse
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from nano_lm.artifacts import atomic_json
from nano_lm.config import load_experiment
from nano_lm.data import sample_batch
from nano_lm.model import GPT
from nano_lm.train import seed_all


def audit(root, device='cpu'):
    torch.set_num_threads(1)
    base, defaults, _ = load_experiment(root, 'E01', 65)
    cfg, training, variant = load_experiment(root, 'E05', 65)
    assert defaults == training and variant['tokenizer'] == 'character'
    assert variant['model_overrides'] == {'activation': 'relu'}
    assert {k for k in cfg.to_dict() if cfg.to_dict()[k] != base.to_dict()[k]} == {'activation'}
    seed_all(42); baseline = GPT(base).to(device)
    seed_all(42); model = GPT(cfg).to(device)
    assert baseline.parameter_count() == model.parameter_count() == 1804800
    initial = {k: torch.equal(v, model.state_dict()[k]) for k, v in baseline.state_dict().items()}
    assert all(initial.values()) and baseline.state_dict().keys() == model.state_dict().keys()
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().numpy().tobytes())
    assert model.position_embedding is None and len(model.blocks) == 4
    assert not any(isinstance(m, nn.GELU) for m in model.modules())
    probes = []
    for i, block in enumerate(model.blocks):
        assert isinstance(block.mlp[1], nn.ReLU)
        assert (block.mlp[0].in_features, block.mlp[0].out_features,
                block.mlp[2].in_features, block.mlp[2].out_features) == (192, 768, 768, 192)
        assert not list(block.mlp[1].parameters()) and not block.attention.rope
        x = torch.tensor([-2., -1., 0., 1., 2.], device=device, requires_grad=True)
        y = block.mlp[1](x); y.sum().backward()
        torch.testing.assert_close(y, torch.tensor([0., 0., 0., 1., 2.], device=device))
        torch.testing.assert_close(x.grad, torch.tensor([0., 0., 0., 1., 1.], device=device))
        assert not torch.allclose(y, F.gelu(x))
        probes.append({'block': i, 'input': x.tolist(), 'relu': y.tolist(),
                       'gradient': x.grad.tolist(), 'gelu': F.gelu(x).tolist()})
    x, y = sample_batch(torch.arange(1000) % 65, 128, 4, torch.Generator().manual_seed(42))
    assert torch.equal(x[:, 1:], y[:, :-1])
    x, y = x.to(device), y.to(device)
    model.eval()
    logits, loss, weights = model(x, y, return_attention=True)
    assert logits.shape == (4, 128, 65) and torch.isfinite(logits).all() and torch.isfinite(loss)
    torch.testing.assert_close(loss, F.cross_entropy(logits.flatten(0, 1), y.flatten()))
    assert all(w.shape == (4, 6, 128, 128) and not torch.count_nonzero(w.triu(1)) for w in weights)
    changed = x.clone(); changed[:, 64:] = (changed[:, 64:] + 1) % 65
    torch.testing.assert_close(model(changed)[0][:, :64], logits[:, :64], rtol=1e-5, atol=1e-6)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(block.mlp[0].weight.grad.norm() > 0 and block.mlp[2].weight.grad.norm() > 0 for block in model.blocks)
    # Backpropagate through each complete MLP and inspect the actual preactivation gradient.
    flow = []
    for i, block in enumerate(model.blocks):
        captures = []
        def capture(module, args):
            args[0].retain_grad(); captures.append(args[0])
        handle = block.mlp[1].register_forward_pre_hook(capture)
        block.mlp(torch.randn(2, 8, 192, device=device, requires_grad=True)).square().sum().backward()
        handle.remove(); pre = captures[0]
        negative = pre < 0
        assert negative.any() and not torch.count_nonzero(pre.grad[negative])
        flow.append({'block': i, 'negative_inputs': int(negative.sum()),
                     'nonzero_negative_input_gradients': int(torch.count_nonzero(pre.grad[negative]))})
    return {'status': 'passed', 'experiment': 'E05', 'device': device, 'parameters': 1804800,
            'model': cfg.to_dict(), 'all_initial_E01_parameter_tensors_equal': True,
            'initial_parameter_sha256': h.hexdigest(), 'matching_state_tensors': len(initial),
            'numerical_activation_probes': probes, 'actual_MLP_gradient_flow': flow,
            'checks': ['sole_activation_override', 'four_ReLU_no_GELU', 'MLP_192_768_192',
                       'no_activation_parameters', 'all_seed42_initial_weights_match_E01',
                       'unchanged_shared_module_shapes', 'numerical_ReLU_not_GELU',
                       'negative_input_zero_gradient', 'actual_MLP_negative_zero_gradient',
                       'shifted_targets', 'finite_forward_manual_CE', 'causal_future_invariance',
                       'finite_backward_all_parameters', 'nonzero_MLP_weight_gradients'],
            'limits': 'Preflight numerical correctness only; not trained activation statistics or language quality.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); atomic_json(a.output, audit(a.root, a.device))
