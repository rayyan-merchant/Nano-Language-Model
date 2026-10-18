"""Prepare a clean-source PRIVATE Vercel Python3.12/CPU function with verified weights."""
import argparse
import hashlib
import json
from pathlib import Path
from .prepare_hf_space import build
from api.hosting import MANIFEST, verify
import shutil


def prepare(destination, artifacts):
    destination = Path(destination)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    verify(Path(artifacts), manifest)
    receipt = build(destination)
    for name in manifest['files']:
        target = destination/'private_models'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(artifacts)/name, target)
    verify(destination/'private_models', manifest)
    # These are deployment metadata files, not edits to the source repository's
    # research requirements or scientific modules. Vercel currently supports 3.12+.
    serving = [line.strip() for line in (destination/'requirements-studio.txt').read_text().splitlines()
               if line.strip() and not line.startswith('#')]
    dependencies = ['torch==2.5.1+cpu', 'numpy==1.26.4', 'tokenizers==0.21.0',
                    ] + serving
    (destination/'requirements.txt').write_text(
        '--extra-index-url https://download.pytorch.org/whl/cpu\n' + '\n'.join(dependencies)+'\n', encoding='utf-8')
    (destination/'pyproject.toml').write_text(
        '[build-system]\nrequires=["setuptools==75.6.0"]\nbuild-backend="setuptools.build_meta"\n'
        '[project]\nname="nanolm-studio-serving"\nversion="0.1.0"\nrequires-python="==3.12.*"\n'
        'dependencies='+json.dumps(dependencies, indent=2)+'\n'
        '[tool.setuptools.packages.find]\nwhere=["src"]\n'
        '[tool.vercel]\nentrypoint="app:app"\n'
        '[[tool.uv.index]]\nname="pytorch-cpu"\nurl="https://download.pytorch.org/whl/cpu"\nexplicit=true\n'
        '[tool.uv.sources]\ntorch={index="pytorch-cpu"}\n', encoding='utf-8')
    (destination/'app.py').write_text('from api.vercel_runtime import create_hosted_app\napp = create_hosted_app()\n', encoding='utf-8')
    (destination/'.python-version').write_text('3.12\n', encoding='utf-8')
    (destination/'vercel.json').write_text(json.dumps({
        '$schema': 'https://openapi.vercel.sh/vercel.json', 'framework': 'fastapi',
        'functions': {'app.py': {'maxDuration': 300, 'includeFiles': '{private_models,configs,experiments,assets,report,api}/**'}}, 'regions': ['bom1']}, indent=2)+'\n', encoding='utf-8')
    # No public/ directory or static route exists; model files are private function assets.
    (destination/'README.md').write_text('# NanoLM Studio Vercel inference backend\n'
        'Private server-function weights, canonical GPT/generation, inference only.\n', encoding='utf-8')
    receipt['target'] = 'Vercel Python3.12 CPU; hosted equivalence pending'
    receipt.pop('weights_in_public_bundle', None)
    receipt['private_function_model_files'] = manifest['files']
    receipt['public_static_model_files'] = False
    receipt['generated_files_sha256'] = {name: hashlib.sha256((destination/name).read_bytes()).hexdigest()
        for name in ['requirements.txt','pyproject.toml','app.py','.python-version','vercel.json','README.md']}
    (destination/'release.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.output, args.artifacts)
    print(json.dumps({'source_commit': receipt['source_commit'], 'target': receipt['target']}))
