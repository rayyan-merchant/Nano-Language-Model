"""Independent audit of canonical E04 learned positions; no optimizer updates."""
import argparse
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from nano_lm.artifacts import atomic_json
from nano_lm.config import load_experiment
from nano_lm.generate import generate
from nano_lm.model import GPT


def audit(root, device='cpu'):
    torch.set_num_threads(1)
    baseline,defaults,_=load_experiment(root,'E01',65)
    config,training,variant=load_experiment(root,'E04',65)
    assert defaults==training and variant['model_overrides']=={'position':'learned'}
    assert {k for k in config.to_dict() if config.to_dict()[k]!=baseline.to_dict()[k]}=={'position'}
    torch.manual_seed(42);a=GPT(baseline).to(device)
    torch.manual_seed(42);b=GPT(config).to(device)
    assert a.parameter_count()==1804800 and b.parameter_count()==1829376
    assert b.position_embedding.weight.shape==(128,192)
    sa,sb=a.state_dict(),b.state_dict()
    assert set(sb)-set(sa)=={'position_embedding.weight'} and not set(sa)-set(sb)
    assert all(sa[k].shape==sb[k].shape for k in sa)
    assert torch.equal(sa['token_embedding.weight'],sb['token_embedding.weight'])
    later_weights_equal=all(torch.equal(sa[k],sb[k]) for k in sa)
    assert not later_weights_equal  # Additional embedding initialization consumes RNG.
    assert all(not block.attention.rope for block in b.blocks)
    b.eval();ids=torch.randint(0,65,(2,128),device=device)
    targets=torch.randint(0,65,(2,128),device=device)
    observed=[];indices=[]
    handle=b.dropout.register_forward_pre_hook(lambda module,args:observed.append(args[0].detach().clone()))
    pos_handle=b.position_embedding.register_forward_pre_hook(lambda module,args:indices.append(args[0].detach().clone()))
    with patch('nano_lm.attention.apply_rope',side_effect=AssertionError('RoPE must be inactive')):
        logits,loss,attentions=b(ids,targets,return_attention=True)
    handle.remove();pos_handle.remove()
    torch.testing.assert_close(indices[0],torch.arange(128,device=device))
    torch.testing.assert_close(observed[0],b.token_embedding(ids)+b.position_embedding(torch.arange(128,device=device)))
    assert logits.shape==(2,128,65) and torch.isfinite(logits).all() and torch.isfinite(loss)
    assert len(attentions)==4
    assert all(w.shape==(2,6,128,128) and torch.count_nonzero(w.triu(1))==0 for w in attentions)
    torch.testing.assert_close(loss,F.cross_entropy(logits.flatten(0,1),targets.flatten()))
    b.sinusoidal.fill_(float('nan'));torch.testing.assert_close(b(ids)[0],logits)
    changed=ids.clone();changed[:,64:]=(changed[:,64:]+1)%65
    torch.testing.assert_close(b(changed)[0][:,:64],logits[:,:64],rtol=1e-5,atol=1e-6)
    loss.backward();grad=b.position_embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.all(grad.norm(dim=1)>0)
    for block in b.blocks:
        for projection in (block.attention.q,block.attention.k,block.attention.v):
            assert torch.isfinite(projection.weight.grad).all() and projection.weight.grad.norm()>0
    prompt=torch.randint(0,65,(1,139),device=device)
    expected=b(prompt[:,-128:])[0][:,-1].argmax(-1)
    torch.testing.assert_close(generate(b,prompt,1,greedy=True)[:,-1],expected)
    return {'status':'passed','experiment':'E04','device':device,'parameters':1829376,'additional_parameters':24576,
            'model':config.to_dict(),'token_initial_weights_match_E01':True,'all_shared_initial_weights_match_E01':later_weights_equal,
            'initialization_limit':'Canonical extra embedding consumes seeded RNG before later layers; shared later weights differ. No custom initialization introduced.',
            'checks':['only_position_factor','embedding_shape_parameter_count_delta','shared_parameter_shapes',
                      'token_plus_learned_positions','local_indices_0_to_127','no_RoPE','sinusoidal_unused',
                      'attention_shapes_causal_mask','finite_forward_manual_CE','future_token_invariance',
                      'finite_nonzero_all_position_gradients','QKV_gradients','cropped_generation_consistency'],
            'limit':'Fixed context correctness only; no extrapolation or language quality proof'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();atomic_json(a.output,audit(a.root,a.device))
