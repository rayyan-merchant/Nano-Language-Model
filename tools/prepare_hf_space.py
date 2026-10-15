"""Build a public, credential/weight-free Space bundle from clean committed code."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def git(*args):
    return subprocess.check_output(['git', *args])


def build(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Use a new immutable Space source destination')
    if git('status', '--porcelain').strip():
        raise ValueError('Commit and verify source changes before preparing deployment')
    commit = git('rev-parse', 'HEAD').decode().strip()
    prefixes = ('src/nano_lm/', 'api/', 'configs/', 'experiments/configs/', 'assets/tokenizers/')
    exact = {
        'requirements.txt', 'requirements-studio.txt', 'requirements-hosting.txt', 'pyproject.toml',
        'report/pilots/E02_seed42/evidence/artifact_manifest.json',
        'report/balanced/evidence/research/artifact_manifest.json',
        'report/final/evidence/research/artifact_manifest.json',
        'report/pilots/E02_seed42/summary.json', 'report/balanced/summary.json', 'report/final/summary.json',
    }
    files = {}
    for name in git('ls-tree', '-r', '--name-only', commit).decode().splitlines():
        if name in exact or name.startswith(prefixes):
            if Path(name).name.startswith('.env'):
                continue
            if Path(name).suffix in {'.pt', '.safetensors', '.zip', '.key', '.pem'}:
                raise ValueError('Private artifact in public source selection')
            data = git('show', f'{commit}:{name}')
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
    required = {'api/hosting.py', 'api/hosting_manifest.json', 'api/Dockerfile.hf'} | exact
    if not required.issubset(files):
        raise ValueError('Incomplete committed hosting source')
    (destination/'Dockerfile').write_bytes((destination/'api/Dockerfile.hf').read_bytes())
    (destination/'README.md').write_text(
        '---\ntitle: NanoLM Studio API\nemoji: 📖\ncolorFrom: purple\ncolorTo: yellow\n'
        'sdk: docker\napp_port: 8000\n---\n\n'
        '# NanoLM Studio inference API\n\n'
        f'Pinned source: `{commit}`. Free CPU research demonstration.\n'
        'Weights remain in a separate private repository. No training or original-test assessment.\n'
        'One bounded inference worker; busy requests receive 429.\n', encoding='utf-8')
    receipt = {'source_commit': commit, 'source_files_sha256': files,
               'weights_in_public_bundle': False, 'secrets_in_public_bundle': False}
    (destination/'release.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.output)
    print(json.dumps({'source_commit': receipt['source_commit'], 'files': len(receipt['source_files_sha256'])}))
