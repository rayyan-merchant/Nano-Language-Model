import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from nano_lm.storage import verify_snapshot
from nano_lm.checkpoint import load_for_inference
from nano_lm.generate import generate
from nano_lm.plot_run import plot_checkpoint
import torch

root = Path.cwd()
archive = root / 'outputs/logs/gpu.zip'
expected = '9966c4bbc31d3479797ebace698c98d949200460d553d20c69586ad5877d4b03'
actual = hashlib.sha256(archive.read_bytes()).hexdigest()
assert actual == expected
destination = root / 'outputs/logs/m4-final-gpu'
destination.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name in z.namelist():
        target = (destination / name).resolve()
        assert target.is_relative_to(destination.resolve())
    z.extractall(destination)
snapshots = sorted(destination.glob('exports/*/export_manifest.json'))
for manifest in snapshots:
    verify_snapshot(manifest.parent)
result = json.loads((destination / 'verification.json').read_text(encoding='utf-8'))
assert all(result['exact_resume'].values()) and result['history_equal']
run = next(p for p in (destination / 'runs').iterdir() if p.name.startswith('E02'))
checkpoint = run / 'checkpoints/step_000002.pt'
model, tokenizer = load_for_inference(checkpoint)
sample = generate(model, torch.tensor([tokenizer.encode('To be')]), 8, greedy=True)
plot_checkpoint(checkpoint, destination / 'smoke_figures')
shutil.copyfile(destination / 'verification.json', root / 'docs/evidence/gpu_verification.json')
gpu_meta = json.loads((run / 'metadata.json').read_text())
cpu_run = next(p for p in (root / 'outputs/logs/m4-final-cpu/runs').iterdir() if p.name.startswith('E02'))
cpu_meta = json.loads((cpu_run / 'metadata.json').read_text())
assert cpu_meta['environment']['code_hash'] == gpu_meta['environment']['code_hash']
record = {'archive_sha256': actual, 'snapshots_verified': len(snapshots),
          'cpu_load_gpu_checkpoint': True, 'sample': tokenizer.decode(sample[0].tolist()),
          'label': '2-step smoke, not a study result',
          'matching_cpu_gpu_source_hash': gpu_meta['environment']['code_hash'],
          'gpu_environment': gpu_meta['environment'],
          'runtime_release': 'Connector release_on_success after verified archive publication',
          'limitation': 'Kaggle download and mounted Google Drive not tested'}
(root / 'docs/evidence/gpu_export_validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='gpu_environment'},indent=2))
