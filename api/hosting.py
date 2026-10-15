"""HF Space startup: fetch pinned private inference files, verify, then serve.

No credentials or model bytes belong in the public Space source/image. The SDK
handles authenticated redirects; application code never logs tokens or signed URLs.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

MANIFEST = Path(__file__).with_name('hosting_manifest.json')


def verify(directory, manifest):
    for name, expected in manifest['files'].items():
        path = directory / name
        if path.is_symlink() or path.stat().st_size != expected['bytes']:
            raise ValueError('Inference package size/type mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('Inference package hash mismatch')


def prepare(repo, revision, root, token, download=None):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('Expected an explicit Hub owner/repository')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Pin an immutable Hub commit SHA, not a branch')
    if not token:
        raise ValueError('Private model repository read secret is required')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    root = Path(root)
    destination = root / revision
    if destination.exists():
        verify(destination, manifest)
        return destination
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.models-', dir=root))
    try:
        for name, expected in manifest['files'].items():
            cached = Path(download(repo_id=repo, filename=name, revision=revision, token=token))
            if cached.stat().st_size != expected['bytes']:
                raise ValueError('Downloaded inference file size mismatch')
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
        verify(staging, manifest)
        staging.rename(destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    try:
        destination = prepare(os.environ['NANOLM_MODEL_REPO'],
                              os.environ['NANOLM_MODEL_REVISION'],
                              os.getenv('NANOLM_MODEL_CACHE', '/home/user/models'),
                              os.environ.pop('NANOLM_MODEL_READ_TOKEN'))
        os.environ['NANOLM_ARTIFACTS'] = str(destination)
        # Fully load and verify all identities before public serving starts.
        import torch
        torch.set_num_threads(int(os.getenv('NANOLM_CPU_THREADS', '1')))
        from .registry import Registry
        registry = Registry(destination)
        for model_id in registry.entries:
            registry.load(model_id)
        from .main import create_app
        import uvicorn
        uvicorn.run(create_app(registry), host='0.0.0.0', port=8000,
                    proxy_headers=False, workers=1, limit_concurrency=32)
    except Exception:
        # SDK exceptions can contain signed URLs. Do not expose their text.
        raise SystemExit('Verified private model startup failed; check deployment configuration') from None


if __name__ == '__main__':
    main()
