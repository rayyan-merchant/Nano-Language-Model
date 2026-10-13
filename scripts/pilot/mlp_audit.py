"""E10 MLP numerical audit; development evidence, no optimizer updates."""
import argparse
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from nano_lm.artifacts import atomic_json
from nano_lm.config import load_experiment
from nano_lm.generate import generate
from nano_lm.model import GPT


def audit(root, device='cpu'):
    torch.set_num_threads(1)
    baseline, defaults, _ = load_experiment(root, 'E01', 65)
    config, training, variant = load_experiment(root, 'E10', 65)
    assert training == defaults and variant['model_overrides'] == {'mlp_expansion': 2}
    assert {k for k in config.to_dict() if config.to_dict()[k] != baseline.to_dict()[k]} == {'mlp_expansion'}
    torch.manual_seed(42); a = GPT(baseline).to(device)
    torch.manual_seed(42); b = GPT(config).to(device)
    torch.manual_seed(42); repeat = GPT(config).to(device)
    assert all(torch.equal(v, repeat.state_dict()[k]) for k,v in b.state_dict().items())
    assert a.parameter_count() == 1804800 and b.parameter_count() == 1213440
    shared = {k: torch.equal(v,a.state_dict()[k]) for k,v in b.state_dict().items() if v.shape == a.state_dict()[k].shape}
    assert b.position_embedding is None and config.position == 'sinusoidal'
    assert len(b.blocks) == 4 and config.n_heads == 6 and config.d_model == 192
    assert len({p.data_ptr() for block in b.blocks for p in block.parameters()}) == sum(len(list(block.parameters())) for block in b.blocks)
    b.eval(); tokens = torch.randint(0,65,(2,129),device=device)
    ids, targets = tokens[:,:-1],tokens[:,1:]
    x = b.token_embedding(ids)+b.sinusoidal[:128]
    counts=[]
    for block in b.blocks:
        first,activation,last = block.mlp
        assert isinstance(activation,nn.GELU)
        assert (first.in_features,first.out_features,last.in_features,last.out_features)==(192,384,384,192)
        count=sum(p.numel() for p in block.mlp.parameters());assert count==148032
        counts.append(count)
        assert block.dropout.p==block.attention.dropout.p==b.dropout.p==0.1 and not block.attention.rope
        assert isinstance(block.norm1,nn.LayerNorm) and isinstance(block.norm2,nn.LayerNorm)
        normalized=block.norm1(x);att=block.attention
        q,k,v=[layer(normalized).reshape(2,128,6,32).transpose(1,2) for layer in (att.q,att.k,att.v)]
        scores=q@k.transpose(-2,-1)/math.sqrt(32)
        mask=torch.ones(128,128,device=device,dtype=torch.bool).triu(1)
        probabilities=scores.masked_fill(mask,float('-inf')).softmax(-1)
        expected_att=F.linear((probabilities@v).transpose(1,2).reshape(2,128,192),att.output.weight,att.output.bias)
        observed_att,weights=att(normalized)
        torch.testing.assert_close(observed_att,expected_att);torch.testing.assert_close(weights,probabilities)
        y=x+expected_att;z=block.norm2(y)
        expected_mlp=F.linear(F.gelu(F.linear(z,first.weight,first.bias)),last.weight,last.bias)
        assert expected_mlp.shape==(2,128,192)
        torch.testing.assert_close(block.mlp(z),expected_mlp)
        expected=y+expected_mlp;observed,_=block(x);torch.testing.assert_close(observed,expected);x=observed
    logits,loss,attention=b(ids,targets,return_attention=True)
    torch.testing.assert_close(logits,b.head(b.norm(x)))
    assert logits.shape==(2,128,65) and torch.isfinite(logits).all() and torch.isfinite(loss)
    assert all(w.shape==(2,6,128,128) and torch.count_nonzero(w.triu(1))==0 for w in attention)
    torch.testing.assert_close(loss,F.cross_entropy(logits.flatten(0,1),targets.flatten()))
    changed=ids.clone();changed[:,64:]=(changed[:,64:]+1)%65
    torch.testing.assert_close(b(changed)[0][:,:64],logits[:,:64],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(b(ids)[0],logits)
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())
    grads=[]
    for i,block in enumerate(b.blocks):
        values=[float(layer.weight.grad.norm()) for layer in [block.mlp[0],block.mlp[2]]]
        assert all(v>0 for v in values);grads.append({'block':i,'projection_weight_gradient_norms':values})
    b.train();assert not torch.equal(b(ids)[0],b(ids)[0]);b.eval()
    prompt=torch.randint(0,65,(1,139),device=device)
    torch.testing.assert_close(generate(b,prompt,1,greedy=True)[:,-1],b(prompt[:,-128:])[0][:,-1].argmax(-1))
    return {'status':'passed','experiment':'E10','device':device,'model':config.to_dict(),'parameters':b.parameter_count(),'parameter_reduction':591360,'MLP_parameters_per_block':counts,'projection_weight_gradients':grads,'seed42_compatible_shape_parameter_equality':shared,'checks':['sole_MLP_override_training_unchanged','repeat_seed42_initialization','four_independent_blocks','192_384_192_projections','GELU_manual_MLP_reference','MLP_and_total_parameter_counts','preLN_residual_reference','manual_sqrt32_causal_attention','sinusoidal_no_RoPE_or_learned','finite_shifted_CE_forward','future_token_invariance','finite_all_gradients_both_MLP_projections','train_eval_dropout_01','cropped_generation'],'limit':'Development gradients establish connectivity, not longitudinal research gradient behavior; changed tensor shapes consume initialization RNG differently.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();atomic_json(a.output,audit(a.root,a.device))
