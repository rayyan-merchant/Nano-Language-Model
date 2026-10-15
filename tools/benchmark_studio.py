"""Actual local serving measurements, never model-quality or training evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import platform
from pathlib import Path
import time
import psutil
import torch
from fastapi.testclient import TestClient
from api.main import create_app
from api.registry import Registry, ModelUnavailable
from api.schemas import GenerateRequest
from api.inference import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    registry = Registry(args.artifacts)
    process = psutil.Process()
    rows = []
    for model_id in registry.entries:
        try:
            registry.load(model_id)
        except ModelUnavailable:
            rows.append({'model_id': model_id, 'status': 'unavailable'});continue
        run(registry, GenerateRequest(model_id=model_id, prompt='ROMEO:\n', max_new_tokens=8), 'warmup')
        for length in [64, 128, 256]:
            for repeat in range(3):
                result = run(registry, GenerateRequest(model_id=model_id, prompt='ROMEO:\n', max_new_tokens=length), 'benchmark')
                rows.append({'model_id': model_id, 'status': 'measured', 'new_tokens': length,
                             'repeat': repeat, 'seconds': result.latency_seconds, 'tokens_per_second': length/result.latency_seconds,
                             'rss_bytes': process.memory_info().rss})
    with TestClient(create_app(registry, rate_limit=1000)) as client:
        start = time.perf_counter()
        comparison = client.post('/v1/compare', json={'prompt': 'ROMEO:\n', 'max_new_tokens': 128}).json()
        compare_wall = time.perf_counter()-start
        with ThreadPoolExecutor(max_workers=2) as executor:
            statuses = list(executor.map(lambda _: client.post('/v1/generate', json={'prompt': 'ROMEO:\n', 'max_new_tokens': 256}).status_code, range(2)))
    result = {'label': 'Actual local CPU serving benchmark, not research training',
              'python': platform.python_version(), 'torch': torch.__version__, 'cpu': platform.processor(),
              'logical_cpus': psutil.cpu_count(), 'threads': 1, 'load_seconds': registry.load_seconds,
              'measurements': rows, 'compare_wall_seconds': compare_wall, 'comparison': comparison,
              'concurrent_request_statuses': statuses, 'rss_after_bytes': process.memory_info().rss,
              'peak_working_set_bytes': getattr(process.memory_info(), 'peak_wset', None)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'load_seconds':registry.load_seconds,'rows':len(rows),'compare_wall_seconds':compare_wall,'concurrent_statuses':statuses}))


if __name__ == '__main__':
    main()
