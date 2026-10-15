"""Explicit genuine-model integration receipt; never falls back to test fixtures."""
import argparse
import json
from pathlib import Path
import torch
from fastapi.testclient import TestClient
from nano_lm.generate import generate
from api.main import create_app
from api.registry import Registry,ModelUnavailable


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    registry=Registry(args.artifacts);torch.set_num_threads(1)
    rows=[]
    with TestClient(create_app(registry,rate_limit=1000)) as client:
        for model_id in registry.entries:
            try:model=registry.load(model_id)
            except ModelUnavailable:rows.append({'id':model_id,'status':'unavailable'});continue
            for greedy in [True,False]:
                prompt='ROMEO:\n'
                body={'model_id':model_id,'prompt':prompt,'max_new_tokens':32,'greedy':greedy,'seed':42,'temperature':.7}
                response=client.post('/v1/generate',json=body);assert response.status_code==200
                result=response.json()
                ids=torch.tensor([registry.tokenizer.encode(prompt)])
                raw=generate(model,ids,32,.7,greedy,42)
                assert result['full_output']==registry.tokenizer.decode(raw[0].tolist())
                assert result['generated_tokens']==32 and result['checkpoint_step']==registry.entries[model_id]['checkpoint_step']
                rows.append({'id':model_id,'status':'passed','greedy':greedy,'canonical_decoding_equal':True,'actual_response':result})
        compare=client.post('/v1/compare',json={'prompt':'ROMEO:\n','max_new_tokens':32}).json()
        assert len(compare['results'])==3
        readiness=client.get('/ready')
    receipt={'scope':'Genuine locally hash-verified inference-only packages; zero training/test likelihood',
             'three_model_acceptance':all(r['status']=='passed' for r in rows),
             'ready_status':readiness.status_code,'models':rows,'comparison':compare}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'three_model_acceptance':receipt['three_model_acceptance'],'ready_status':receipt['ready_status'],'cases':len(rows)}))


if __name__=='__main__':main()
