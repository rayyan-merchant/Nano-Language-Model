"""Pure B01 fixed-grid reproduction; never optimize or evaluate test likelihood."""
import json
from pathlib import Path
import torch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.checkpoint import load_for_inference
from nano_lm.generate import generate
from nano_lm.train import seed_all
from scripts.final.reproduce_generation import generated_bytes
from scripts.balanced.protocol import MODEL, PARAMETERS, BPE


def reproduce(root,original,out,device='cuda'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);seed_all(42)
    completion=json.loads((original/'completion.json').read_text(encoding='utf-8'))
    grid=json.loads((original/'analysis/generation.json').read_text(encoding='utf-8'))
    assert grid['protocol']==json.loads((root/'configs/generation_protocol.json').read_text(encoding='utf-8'))
    assert len(grid['samples'])==100
    run=(original/completion['final_checkpoint']).parent.parent;records=[]
    for policy in ['best','final']:
        samples=[s for s in grid['samples'] if s['checkpoint_policy']==policy]
        assert len(samples)==50 and len({s['checkpoint'] for s in samples})==1
        cp=run/'checkpoints'/samples[0]['checkpoint'];model,tok=load_for_inference(cp);model.to(device)
        assert model.config.to_dict()==MODEL and model.parameter_count()==PARAMETERS and tok.checksum==BPE
        for slot in samples:
            assert file_hash(cp)==slot['checkpoint_sha256'] and slot['max_new_tokens']==256
            prompt=tok.encode(slot['prompt'])
            ids=generate(model,torch.tensor([prompt],device=device),256,greedy=slot['greedy'],
                         temperature=slot['temperature'],seed=slot['seed'])[0].tolist()
            assert tok.decode(ids)==slot['text'] and len(ids)==len(prompt)+256
            raw=generated_bytes(tok,ids[len(prompt):])
            try:raw.decode('utf-8');valid=True
            except UnicodeDecodeError:valid=False
            records.append({'policy':policy,'prompt':slot['prompt'],'greedy':slot['greedy'],
                'temperature':slot['temperature'],'seed':slot['seed'],'checkpoint_sha256':file_hash(cp),
                'generated_token_ids':ids[len(prompt):],'strict_UTF8_valid':valid,
                'original_generation_exactly_reproduced':True})
        del model;torch.cuda.empty_cache() if device=='cuda' else None
    result={'status':'all100_B01_fixed_generations_exactly_reproduced','samples':records,
        'generation_file_sha256':file_hash(original/'analysis/generation.json'),
        'additional_optimizer_updates':0,'test_likelihood_evaluations':0,
        'strict_UTF8_invalid_outputs':sum(not r['strict_UTF8_valid'] for r in records)}
    atomic_json(out/'generation_reproduction.json',result)
    return {'status':result['status'],'slots':100,'additional_optimizer_updates':0,
            'receipt':str(out/'generation_reproduction.json'),'strict_UTF8_invalid_outputs':result['strict_UTF8_invalid_outputs']}
