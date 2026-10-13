"""Independent eight-head math and canonical E07 model audit; no updates."""
import argparse
import hashlib
import math
from pathlib import Path
import torch
from torch import nn
from nano_lm.artifacts import atomic_json
from nano_lm.attention import split_heads, merge_heads, scaled_attention
from nano_lm.config import load_experiment
from nano_lm.data import sample_batch
from nano_lm.generate import generate
from nano_lm.model import GPT
from nano_lm.train import seed_all


def audit(root, device='cpu'):
    torch.set_num_threads(1)
    base, defaults, _ = load_experiment(root, 'E01', 65)
    cfg, training, variant = load_experiment(root, 'E07', 65)
    assert defaults == training and variant['tokenizer'] == 'character'
    assert variant['model_overrides'] == {'n_heads': 8}
    assert {k for k in cfg.to_dict() if cfg.to_dict()[k] != base.to_dict()[k]} == {'n_heads'}
    assert cfg.d_model == 192 and cfg.n_heads == 8 and cfg.d_model % cfg.n_heads == 0
    seed_all(42); baseline = GPT(base).to(device)
    seed_all(42); model = GPT(cfg).to(device)
    assert baseline.parameter_count() == model.parameter_count() == 1804800
    assert baseline.state_dict().keys() == model.state_dict().keys()
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in baseline.state_dict().items())
    h = hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    assert model.position_embedding is None and cfg.position == 'sinusoidal' and cfg.activation == 'gelu'
    assert len(model.blocks) == 4 and not any(isinstance(m,nn.ReLU) for m in model.modules())
    probes=[]
    # Deterministic real projection values distinguish all requested scale denominators.
    x=torch.linspace(-2,2,2*7*192,device=device).reshape(2,7,192)
    for i,b in enumerate(model.blocks):
        a=b.attention
        assert a.heads==8 and not a.rope and a.dropout.p==.1
        assert isinstance(b.mlp[1],nn.GELU)
        assert (b.mlp[0].in_features,b.mlp[0].out_features,b.mlp[2].in_features,b.mlp[2].out_features)==(192,768,768,192)
        a.eval()
        q,k,v=[split_heads(layer(x),8) for layer in (a.q,a.k,a.v)]
        assert q.shape==k.shape==v.shape==(2,8,7,24)
        raw=q@k.transpose(-2,-1);mask=torch.ones(7,7,device=device,dtype=torch.bool).triu(1)
        probabilities=(raw/math.sqrt(24)).masked_fill(mask,float('-inf')).softmax(-1)
        attended,actual=scaled_attention(q,k,v,dropout=a.dropout)
        torch.testing.assert_close(actual,probabilities)
        torch.testing.assert_close(actual.sum(-1),torch.ones(2,8,7,device=device))
        assert not torch.count_nonzero(actual.triu(1)) and merge_heads(attended).shape==(2,7,192)
        expected=a.output(merge_heads(probabilities@v));output,weights=a(x)
        torch.testing.assert_close(output,expected);torch.testing.assert_close(weights,actual)
        differences={str(d):float((actual-(raw/math.sqrt(d)).masked_fill(mask,float('-inf')).softmax(-1)).abs().max().detach()) for d in [32,192]}
        assert all(value>1e-4 for value in differences.values())
        a.train();seed_all(42)
        dropped=a.dropout(probabilities)
        assert (dropped==0).sum()>(probabilities==0).sum()
        positive=dropped>0
        torch.testing.assert_close(dropped[positive],probabilities[positive]/.9)
        a.eval();torch.testing.assert_close(a.dropout(probabilities),probabilities)
        probes.append({'block':i,'qkv_shape':list(q.shape),'logits_shape':list(raw.shape),'scale_divisor':math.sqrt(24),'wrong_scale_max_probability_difference':differences,'merged_shape':list(merge_heads(attended).shape),'dropout_train_eval_passed':True})
    ids,targets=sample_batch(torch.arange(1000)%65,128,4,torch.Generator().manual_seed(42))
    assert torch.equal(ids[:,1:],targets[:,:-1])
    ids,targets=ids.to(device),targets.to(device);model.eval()
    logits,loss,weights=model(ids,targets,return_attention=True)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    torch.testing.assert_close(loss,nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten()))
    assert all(w.shape==(4,8,128,128) and not torch.count_nonzero(w.triu(1)) for w in weights)
    changed=ids.clone();changed[:,64:]=(changed[:,64:]+1)%65
    torch.testing.assert_close(model(changed)[0][:,:64],logits[:,:64],rtol=1e-5,atol=1e-6)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    gradients=[]
    for i,b in enumerate(model.blocks):
        row={'block':i}
        for name in ['q','k','v','output']:
            value=float(getattr(b.attention,name).weight.grad.norm());assert value>0
            row[name]=value
        gradients.append(row)
    long_ids=torch.arange(150,device=device).reshape(1,-1)%65
    expected=model(long_ids[:,-128:])[0][:,-1].argmax(-1)
    generated=generate(model,long_ids,1,greedy=True)
    assert torch.equal(generated[:,-1],expected) and not model.training
    return {'status':'passed','experiment':'E07','device':device,'parameters':1804800,'model':cfg.to_dict(),
            'all_initial_E01_parameter_tensors_equal':True,'matching_state_tensors':len(model.state_dict()),
            'initial_parameter_sha256':h.hexdigest(),'numerical_attention_probes':probes,'projection_gradient_norms':gradients,
            'checks':['sole_n_heads_override','eight_heads_dim24','QKV_and_logits_shapes','sqrt24_manual_reference','wrong_sqrt32_sqrt192_distinguished','per_head_key_softmax','causal_mask','dropout_train_eval','merge_width192','GELU_sinusoidal_no_ReLU_RoPE_learned','seed42_initial_E01_tensors_equal','shifted_targets_manual_CE','finite_all_parameter_gradients','nonzero_QKV_output_gradients','future_token_invariance','generation_context_crop128'],
            'limits':'Preflight correctness only; no trained head diversity or causal mechanism inferred.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cpu',choices=['cpu','cuda']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();atomic_json(a.output,audit(a.root,a.device))
