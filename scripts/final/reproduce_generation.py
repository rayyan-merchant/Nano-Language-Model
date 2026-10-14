"""Fresh compatible fixed-grid inference and strict generated-byte diagnostics."""
import argparse
import json
from pathlib import Path
import subprocess
import torch
from nano_lm.artifacts import atomic_json, file_hash
from nano_lm.checkpoint import load_for_inference
from nano_lm.generate import generate
from nano_lm.train import seed_all, environment
from scripts.final.protocol import MODEL, BPE, PARAMETERS


def generated_bytes(tokenizer,ids):
    # Inverse of the canonical ByteLevel alphabet, including split UTF8 bytes.
    visible=list(range(33,127))+list(range(161,173))+list(range(174,256))
    values=list(visible);codes=list(visible);extra=0
    for byte in range(256):
        if byte not in visible:
            values.append(byte);codes.append(256+extra);extra+=1
    reverse=dict(zip(map(chr,codes),values))
    return bytes(reverse[c] for token in ids for c in tokenizer.backend.id_to_token(token))


def reproduce(root,out,helper_commit):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);seed_all(42)
    env=environment(root)
    assert not env['git_dirty'] and env['python']=='3.11.14' and env['torch']=='2.5.1+cu124'
    assert env['cuda']=='12.4' and env['gpu']=='Tesla T4'
    paths=list(Path('/kaggle/input').rglob('completion.json'));assert len(paths)==1
    original=paths[0].parent;completion=json.loads(paths[0].read_text())
    grid=json.loads((original/'analysis/generation.json').read_text());assert len(grid['samples'])==100
    run=(original/completion['final_checkpoint']).parent.parent;records=[]
    for policy in ['best','final']:
        selected=[s for s in grid['samples'] if s['checkpoint_policy']==policy];assert len(selected)==50
        names={s['checkpoint'] for s in selected};assert len(names)==1
        name=next(iter(names));assert Path(name).name==name
        checkpoint=run/'checkpoints'/name;model,tok=load_for_inference(checkpoint);model.to('cuda')
        assert model.config.to_dict()==MODEL and model.parameter_count()==PARAMETERS and tok.checksum==BPE
        for s in selected:
            assert file_hash(checkpoint)==s['checkpoint_sha256'] and s['max_new_tokens']==256
            prompt=tok.encode(s['prompt']);ids=generate(model,torch.tensor([prompt],device='cuda'),256,
                   greedy=s['greedy'],temperature=s['temperature'],seed=s['seed'])[0].tolist()
            assert len(ids)==len(prompt)+256
            decoded=tok.decode(ids);assert decoded==s['text'],'Fixed CUDA generation differs from published source output'
            new_ids=ids[len(prompt):];raw=generated_bytes(tok,new_ids);invalid=None
            try:strict=raw.decode('utf-8')
            except UnicodeDecodeError as e:
                strict=None;invalid={'first_error_start_byte':e.start,'first_error_end_byte':e.end,'reason':e.reason}
            if strict is not None:assert strict==decoded[len(s['prompt']):]
            records.append({'policy':policy,'checkpoint':name,'checkpoint_sha256':s['checkpoint_sha256'],
                            'prompt':s['prompt'],'greedy':s['greedy'],'temperature':s['temperature'],'seed':s['seed'],
                            'max_new_tokens':256,'generated_token_ids':new_ids,'raw_generated_bytes':len(raw),
                            'strict_UTF8_valid':strict is not None,'UTF8_error':invalid,
                            'replacement_characters':decoded[len(s['prompt']):].count('\ufffd'),
                            'original_generation_exactly_reproduced':True})
        del model;torch.cuda.empty_cache()
    receipt={'status':'all100_fixed_CUDA_generations_exactly_reproduced_with_token_and_byte_diagnostics',
             'source_commit':env['git_commit'],'helper_commit':helper_commit,'helper_file_sha256':file_hash(Path(__file__)),
             'code_hash':env['code_hash'],'generation_file_sha256':file_hash(original/'analysis/generation.json'),
             'samples':records,'additional_optimizer_updates':0,
             'strict_UTF8_invalid_outputs':sum(not s['strict_UTF8_valid'] for s in records),
             'scope':'Pure fixed-protocol inference from recovered own best/final models; original outputs unedited; no checkpoint/model selection'}
    atomic_json(out/'generation_reproduction.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--helper-commit',required=True)
    a=p.parse_args();assert len(a.helper_commit)==40 and all(c in '0123456789abcdef' for c in a.helper_commit)
    reproduce(a.root,a.output,a.helper_commit)
