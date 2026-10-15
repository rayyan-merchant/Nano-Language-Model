"""Genuine public-HTTPS qualification against committed local equivalence receipts."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def verify(origin, output, deployment):
    if not origin.startswith('https://') or output.exists():
        raise ValueError('Use an HTTPS deployment and a fresh receipt path')
    rows = []
    def request(path, body=None):
        encoded = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(origin.rstrip('/')+path, data=encoded,
                                     headers={'Content-Type':'application/json'})
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response), time.perf_counter()-start
    health, health_seconds = request('/health')
    ready, ready_seconds = request('/ready')
    assert ready['ready'] and len(ready['models']) == 3
    for model_id in ['e02','b01','f01']:
        metadata = json.loads(Path(f'docs/deployment/evidence/{model_id}-package.json').read_text(encoding='utf-8'))
        for greedy in [True, False]:
            expected = next(c for c in metadata['equivalence']['cases']
                            if c['prompt']=='ROMEO:\n' and c['greedy']==greedy)
            response, elapsed = request('/v1/generate', {'model_id':model_id,'prompt':'ROMEO:\n',
                'max_new_tokens':32,'temperature':0.7,'seed':42,'greedy':greedy})
            assert response['model_id']==model_id and response['checkpoint_step']==metadata['checkpoint_step']
            assert response['generated_tokens']==32 and response['full_output']==expected['decoded']
            rows.append({'model':model_id,'greedy':greedy,'wall_seconds':elapsed,
                         'exact_local_export_decoding_equal':True,'actual_response':response})
    comparison, compare_seconds = request('/v1/compare', {'prompt':'ROMEO:\n','max_new_tokens':32,'seed':42})
    assert len(comparison['results']) == 3 and all(r['status']=='completed' for r in comparison['results'])
    private_checks = []
    for path in [f'/private_models/{m}/{f}' for m in ['e02','b01','f01']
                 for f in ['model.safetensors','metadata.json']] + ['/.env.local','/release.json']:
        try:
            urllib.request.urlopen(origin.rstrip('/')+path, timeout=30).close()
            raise AssertionError('Private deployment file is publicly accessible')
        except urllib.error.HTTPError as error:
            assert error.code==404
            private_checks.append({'path':path,'status':404})
    receipt = {'origin':origin,'deployment':deployment,'scope':'Actual public HTTPS inference; zero training/test likelihood',
               'health':health,'health_seconds':health_seconds,'ready_seconds':ready_seconds,
               'three_models_ready':True,'generation_cases':rows,'comparison':comparison,
               'compare_wall_seconds':compare_seconds,'private_file_checks':private_checks,
               'raw_token_ID_cross_runtime_equality':'Not exposed by HTTP API; exact decoded results compared',
               'all_checks_passed':True}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin',required=True)
    parser.add_argument('--deployment',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    receipt=verify(args.origin,args.output,args.deployment)
    print(json.dumps({'all_checks_passed':True,'cases':len(receipt['generation_cases']),
                      'private_file_checks':len(receipt['private_file_checks'])}))
