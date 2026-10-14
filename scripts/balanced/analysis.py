"""Measured B01 analysis through existing plot/grid utilities, no test reading."""
import json
from pathlib import Path
import time
import torch
import numpy as np
from nano_lm.artifacts import atomic_json, file_hash
from scripts.pilot import analyze as shared
from scripts.final.reproduce_generation import generated_bytes


def analyze(root,run,out,device='cuda',development=False):
    records=[];loads=[]
    original_generate,original_load=shared.generate,shared.load_for_inference
    def observed(model,prompt,maximum,**settings):
        if device=='cuda':torch.cuda.synchronize()
        began=time.perf_counter();result=original_generate(model,prompt,maximum,**settings)
        if device=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-began
        records.append({'generated_token_ids':result[0,prompt.shape[1]:].tolist(),'seconds':seconds})
        return result
    def loaded(path):
        began=time.perf_counter();result=original_load(path)
        loads.append({'checkpoint':Path(path).name,'CPU_deserialize_construct_seconds':time.perf_counter()-began})
        return result
    shared.generate,shared.load_for_inference=observed,loaded
    try:shared.analyze(Path(root),Path(run),Path(out),device,development_check=development)
    finally:shared.generate,shared.load_for_inference=original_generate,original_load
    grid=json.loads((Path(out)/'generation.json').read_text(encoding='utf-8'));assert len(records)==len(grid['samples'])
    tok=original_load(Path(run)/'checkpoints'/grid['samples'][0]['checkpoint'])[1]
    for slot,record in zip(grid['samples'],records):
        raw=generated_bytes(tok,record['generated_token_ids'])
        try:raw.decode('utf-8');valid=True;error=None
        except UnicodeDecodeError as exc:valid=False;error={'start':exc.start,'end':exc.end,'reason':exc.reason}
        record.update(checkpoint_policy=slot['checkpoint_policy'],prompt=slot['prompt'],greedy=slot['greedy'],
            temperature=slot['temperature'],seed=slot['seed'],raw_generated_bytes=len(raw),strict_UTF8_valid=valid,UTF8_error=error)
    atomic_json(Path(out)/'generation_token_diagnostics.json',{'samples':records,'loads':loads,
        'strict_UTF8_invalid_outputs':sum(not r['strict_UTF8_valid'] for r in records),
        'scope':'Original fixed grid only; synchronized generation latency excludes byte diagnostics/decoding and model load. First request per policy is cold; no extra inference or optimizer updates.'})
    if not development:
        history=json.loads((Path(out)/'measured_history.json').read_text(encoding='utf-8'))['training']
        intervals=[]
        for end in range(250,len(history)+1,250):
            norms=np.array([r['gradient_norm'] for r in history[end-250:end]])
            intervals.append({'from':end-249,'through':end,'median':float(np.median(norms)),
                'p95':float(np.quantile(norms,.95)),'maximum':float(norms.max()),'clipped_updates':int((norms>1).sum())})
        atomic_json(Path(out)/'gradient_intervals.json',{'intervals':intervals,
            'total_clipped_updates':sum(r['clipped_updates'] for r in intervals),'observed_updates':len(history)})
