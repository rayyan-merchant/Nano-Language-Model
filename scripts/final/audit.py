"""Independent numerical references for the exact combined F01 architecture."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from nano_lm.artifacts import atomic_json
from nano_lm.bpe import from_artifact, encode_with_bytes
from nano_lm.evaluate import evaluation_tokens, metrics, evaluate
from nano_lm.generate import generate
from nano_lm.model import GPT
from nano_lm.positions import apply_rope
from nano_lm.train import seed_all
from scripts.final.protocol import BPE, PARAMETERS, configuration, parameter_breakdown, provenance


def reference_rotation(x, wrong=False):
    # Independent pair loop, rather than the implementation's stack/flatten formula.
    y = torch.empty_like(x)
    positions = torch.arange(x.shape[-2], device=x.device).float()
    for pair in range(x.shape[-1]//2):
        theta = positions / (10000.0 ** (2*pair/x.shape[-1]))
        c, s = torch.cos(theta).to(x.dtype), torch.sin(theta).to(x.dtype)
        a, b = x[...,2*pair], x[...,2*pair+1]
        y[...,2*pair] = a*c + b*s if wrong else a*c - b*s
        y[...,2*pair+1] = -a*s + b*c if wrong else a*s + b*c
    return y


def audit(root, data, device='cpu', *, configuration_fn=configuration,
          parameter_fn=parameter_breakdown, parameters=PARAMETERS, provenance_fn=provenance):
    root, data = Path(root), Path(data)
    torch.set_num_threads(1); seed_all(42)
    config = configuration_fn(root); model = GPT(config).to(device)
    width, layers, head = config.d_model, config.n_layers, config.d_model//config.n_heads
    seed_all(42); repeat = GPT(config).to(device)
    assert all(torch.equal(v,repeat.state_dict()[k]) for k,v in model.state_dict().items())
    del repeat
    assert model.parameter_count() == parameter_fn()['total'] == parameters
    assert model.head.weight.data_ptr() != model.token_embedding.weight.data_ptr()
    assert model.position_embedding is None and model.config.position == 'rope'
    assert len(model.blocks) == layers
    pointers = [p.data_ptr() for b in model.blocks for p in b.parameters()]
    assert len(set(pointers)) == len(pointers)
    assert model.token_embedding.weight.shape == model.head.weight.shape == (1024,width)
    tok = from_artifact(json.loads((data/'bpe.json').read_text(encoding='utf-8')))
    assert tok.vocab_size == 1024 and tok.checksum == BPE
    for text in ['ROMEO:\n', 'To be, or not to be', 'é € Shakespeare\n']:
        ids, lengths = encode_with_bytes(tok,text)
        assert tok.decode(ids) == text and sum(lengths)==len(text.encode('utf-8'))
    # Byte alphabet's incomplete UTF8 byte must remain observable, not filtered.
    undecodable = [i for i in range(1024) if '\ufffd' in tok.decode([i])]
    assert undecodable
    validation = (data/'validation.txt').read_bytes().decode('utf-8')
    seq, start, byte_lengths, prefix_bytes, total_bytes = evaluation_tokens(tok,validation)
    assert sum(byte_lengths)==55514 and prefix_bytes==256 and total_bytes==55770
    assert len(seq)-start==24386
    analytic = metrics((len(seq)-start)*math.log(1024), len(seq)-start,55514)
    assert math.isclose(analytic['bpb'], (len(seq)-start)*10/55514, rel_tol=1e-12)
    tokens=torch.randint(0,1024,(2,129),device=device);ids,targets=tokens[:,:-1],tokens[:,1:]
    assert torch.equal(ids[:,1:],targets[:,:-1])
    try:model(tokens)
    except ValueError:pass
    else:raise AssertionError('Overlength input accepted')
    model.eval();x=model.token_embedding(ids);incorrect_reference_detected=[]
    for block in model.blocks:
        att=block.attention;first,activation,last=block.mlp
        assert att.heads==6 and att.rope and isinstance(activation,nn.GELU)
        assert first.weight.shape==(4*width,width) and last.weight.shape==(width,4*width)
        assert block.norm1.normalized_shape==block.norm2.normalized_shape==(width,)
        assert block.dropout.p==att.dropout.p==model.dropout.p==0.1
        normalized=block.norm1(x)
        q,k,v=[layer(normalized).reshape(2,128,6,head).transpose(1,2) for layer in (att.q,att.k,att.v)]
        qr,kr=reference_rotation(q),reference_rotation(k)
        # Division versus the canonical float32 inverse-frequency multiplication
        # rounds high-position angles differently. Bound this reference error.
        torch.testing.assert_close(apply_rope(q),qr,rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(apply_rope(k),kr,rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(qr.square().sum(-1),q.square().sum(-1),rtol=1e-5,atol=1e-5)
        mask=torch.ones(128,128,device=device,dtype=torch.bool).triu(1)
        def probabilities(qq,kk,scale):
            return (qq@kk.transpose(-2,-1)/scale).masked_fill(mask,float('-inf')).softmax(-1)
        p=probabilities(qr,kr,math.sqrt(head))
        attended=F.linear((p@v).transpose(1,2).reshape(2,128,width),att.output.weight,att.output.bias)
        observed,weights=att(normalized)
        torch.testing.assert_close(observed,attended,rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(weights,p,rtol=2e-5,atol=2e-5)
        wrong_head=40 if head==32 else 32
        assert not torch.allclose(p, probabilities(qr,kr,math.sqrt(wrong_head)))
        assert not torch.allclose(p, probabilities(reference_rotation(q,True),kr,math.sqrt(head)))
        incorrect_reference_detected.append([f'sqrt{wrong_head}_scale','incorrect_rotary_sign'])
        y=x+attended;normalized_mlp=block.norm2(y)
        reference=y+F.linear(F.gelu(F.linear(normalized_mlp,first.weight,first.bias)),last.weight,last.bias)
        actual,_=block(x);torch.testing.assert_close(actual,reference,rtol=1e-5,atol=4e-6);x=actual
    logits,loss,weights=model(ids,targets,True)
    torch.testing.assert_close(logits,model.head(model.norm(x)),rtol=1e-5,atol=4e-6)
    selected=logits.log_softmax(-1).gather(-1,targets.unsqueeze(-1))
    torch.testing.assert_close(loss,-selected.mean())
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert all(w.shape==(2,6,128,128) and torch.count_nonzero(w.triu(1))==0 for w in weights)
    changed=ids.clone();changed[:,64:]=(changed[:,64:]+1)%1024
    torch.testing.assert_close(model(changed)[0][:,:64],logits[:,:64],rtol=1e-5,atol=2e-6)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in model.parameters())
    gradients=[{'block':i,'q':b.attention.q.weight.grad.norm().item(),
                'MLP_first':b.mlp[0].weight.grad.norm().item(),'MLP_second':b.mlp[2].weight.grad.norm().item()}
               for i,b in enumerate(model.blocks)]
    model.train();assert not torch.equal(model(ids)[0],model(ids)[0]);model.eval()
    prompt=torch.randint(0,1024,(1,145),device=device)
    torch.testing.assert_close(generate(model,prompt,1,greedy=True)[:,-1],model(prompt[:,-128:])[0][:,-1].argmax(-1))
    # A genuinely contextual independent evaluator reference on a tiny real suffix.
    snippet=validation[:320];eval_ids,begin,lens,_,_=evaluation_tokens(tok,snippet)
    nll=0.
    with torch.no_grad():
        for i in range(begin,len(eval_ids)):
            history=torch.tensor([eval_ids[max(0,i-128):i]],device=device)
            lp=model(history)[0][0,-1].log_softmax(-1)
            nll-=float(lp[eval_ids[i]])
    measured=evaluate(model,tok,snippet,batch_size=7)
    assert math.isclose(nll,measured['nll_sum'],rel_tol=1e-6)
    assert measured['scored_bytes']==sum(lens)==64
    return {'status':'passed','device':device,'model':config.to_dict(),'parameters':parameters,
            'parameter_breakdown':parameter_fn(),'provenance':provenance_fn(root),
            'all_eight_block_gradient_norms':gradients,'wrong_reference_detection':incorrect_reference_detected,
            'BPE_checksum':tok.checksum,'BPE_scored_tokens':len(seq)-start,'scored_validation_bytes':55514,
            'incomplete_UTF8_single_token_ids':undecodable,'independent_sliding_reference':measured,
            'checks':'Independent RoPE, scaled causal QKV attention, preLN residual GELU, CE, gradients, masks, dropout, deterministic initialization, context cropping, BPE byte coverage and UTF8 failure retention',
            'optimizer_updates':0,'test_likelihood_evaluations':0}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();atomic_json(a.output,audit(a.root,a.dataset,a.device))
