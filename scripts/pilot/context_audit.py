"""E11 development numerical evidence; never performs optimizer updates."""
import math
import torch
from torch.nn import functional as F
from nano_lm.config import load_experiment
from nano_lm.model import GPT
from nano_lm.generate import generate
from nano_lm.evaluate import evaluate


def token_budget(config, settings):
    """Derive grouping from the resolved model rather than the E01 context."""
    positions = config.context_length * settings.micro_batch_size
    if settings.tokens_per_update % positions:
        raise ValueError('Token budget must divide by resolved microbatch positions')
    accumulation = settings.tokens_per_update // positions
    return {'context_length': config.context_length, 'micro_batch_size': settings.micro_batch_size,
            'accumulation': accumulation, 'sampled_windows_per_update': accumulation * settings.micro_batch_size,
            'predicted_positions_per_update': positions * accumulation}


def audit(root, device='cpu'):
    torch.set_num_threads(1)
    base, defaults, _ = load_experiment(root, 'E01', 65)
    config, training, variant = load_experiment(root, 'E11', 65)
    assert training == defaults and variant['model_overrides'] == {'context_length': 64}
    assert {k for k in config.to_dict() if config.to_dict()[k] != base.to_dict()[k]} == {'context_length'}
    torch.manual_seed(42); baseline = GPT(base).to(device)
    torch.manual_seed(42); model = GPT(config).to(device)
    assert model.parameter_count() == baseline.parameter_count() == 1804800
    assert all(torch.equal(v, baseline.state_dict()[k]) for k,v in model.state_dict().items())
    assert model.sinusoidal.shape == (64,192) and model.position_embedding is None
    torch.testing.assert_close(model.sinusoidal,baseline.sinusoidal[:64])
    assert config.position == 'sinusoidal' and config.activation == 'gelu'
    assert all(not b.attention.rope and b.dropout.p == b.attention.dropout.p == .1 for b in model.blocks)
    tokens=torch.randint(0,65,(2,65),device=device); ids,targets=tokens[:,:64],tokens[:,1:]
    assert torch.equal(ids[:,1:], targets[:,:-1])
    model.eval(); logits,loss,weights=model(ids,targets,True)
    assert logits.shape == (2,64,65) and torch.isfinite(logits).all() and torch.isfinite(loss)
    torch.testing.assert_close(loss,F.cross_entropy(logits.flatten(0,1),targets.flatten()))
    assert all(w.shape == (2,6,64,64) and torch.count_nonzero(w.triu(1)) == 0 for w in weights)
    att=model.blocks[0].attention; x=model.blocks[0].norm1(model.token_embedding(ids)+model.sinusoidal)
    q,k,v=[layer(x).reshape(2,64,6,32).transpose(1,2) for layer in (att.q,att.k,att.v)]
    scores=q@k.transpose(-2,-1)/math.sqrt(32)
    expected=scores.masked_fill(torch.ones(64,64,device=device,dtype=torch.bool).triu(1),float('-inf')).softmax(-1)
    output,probabilities=att(x)
    torch.testing.assert_close(probabilities,expected)
    torch.testing.assert_close(output,att.output((expected@v).transpose(1,2).reshape(2,64,192)))
    changed=ids.clone();changed[:,32:]=(changed[:,32:]+1)%65
    torch.testing.assert_close(model(changed)[0][:,:32],logits[:,:32],atol=1e-6,rtol=1e-5)
    try: model(tokens)
    except ValueError: pass
    else: raise AssertionError('Overlength input accepted')
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in model.parameters())
    model.train();assert not torch.equal(model(ids)[0],model(ids)[0]);model.eval()
    prompt=torch.randint(0,65,(1,140),device=device)
    seen=[]
    handle=model.register_forward_pre_hook(lambda m,args:seen.append(args[0].shape[1]))
    generated=generate(model,prompt,3,greedy=True);handle.remove()
    assert seen == [64]*3
    assert torch.equal(generated[:,140],model(prompt[:,-64:])[0][:,-1].argmax(-1))
    # Independent sequential reference checks the evaluator's context and aggregation.
    class CharacterFixture:
        def encode(self,text): return [ord(c)-65 for c in text]
        def decode(self,ids): return "".join(chr(i+65) for i in ids)
        def byte_lengths(self,ids): return [1]*len(ids)
    text='ABCDE'*60; tokenizer=CharacterFixture()
    evaluated=evaluate(model,tokenizer,text,conditioning_bytes=256,batch_size=7)
    sequential=sum(F.cross_entropy(model(torch.tensor([tokenizer.encode(text[i-64:i])],device=device))[0][:,-1],
                                  torch.tensor([ord(text[i])-65],device=device),reduction='sum').item()
                   for i in range(256,len(text)))
    assert evaluated['context_length']==64 and evaluated['conditioning_bytes']==256 and evaluated['scored_bytes']==44
    assert math.isclose(evaluated['nll_sum'],sequential,rel_tol=1e-6)
    assert math.isclose(evaluated['bpb'],evaluated['loss']/math.log(2),rel_tol=1e-12)
    return {'status':'passed','experiment':'E11','device':device,'parameters':1804800,'model':config.to_dict(),
            'seed42_initial_trainable_weights_equal_E01':True,'evaluation_reference':evaluated,
            'checks':['sole_context_override','equal_initial_trainable_weights','sinusoidal_prefix64',
                      'shifted_targets64','finite_forward_CE','causal_masks64','manual_sqrt32_attention',
                      'future_invariance','reject_overlength','all_parameter_gradients','dropout_train_eval',
                      'generation_crop64','independent_sliding_reference64','conditional_byte_BPB'],
            'limitations':'Synthetic numerical fixture is development evidence, not research data; full real-data validation is a separate gate.'}
