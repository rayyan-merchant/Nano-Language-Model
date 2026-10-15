"""Publish only verified inference files to the intended PRIVATE HF repository.

Requires the operator's private local HF write login. No token arguments or output.
The read-back receipt is required before configuring a public Space.
"""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from api.hosting import MANIFEST, verify


def publish(account, repo, artifacts, receipt):
    if Path(receipt).exists():
        raise FileExistsError('Use a fresh publication receipt')
    api = HfApi()
    if api.whoami()['name'] != account or not repo.startswith(account+'/'):
        raise ValueError('Authenticated Hub identity/namespace mismatch')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    artifacts = Path(artifacts)
    verify(artifacts, manifest)
    api.create_repo(repo, private=True, exist_ok=True)
    info = api.repo_info(repo)
    if not info.private:
        raise ValueError('Refusing to upload weights to a public repository')
    commit = api.create_commit(repo, parent_commit=info.sha,
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(artifacts/name))
                    for name in manifest['files']],
        commit_message='Publish hash-verified selected NanoLM Studio inference packages')
    # Independent authenticated retrieval at the immutable remote revision.
    import tempfile
    import shutil
    with tempfile.TemporaryDirectory(prefix='nanolm-readback-') as temporary:
        root = Path(temporary)
        for name in manifest['files']:
            downloaded = hf_hub_download(repo, name, revision=commit.oid, force_download=True)
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(downloaded, target)
        verify(root, manifest)
    if not api.repo_info(repo, revision=commit.oid).private:
        raise ValueError('Private publication verification failed')
    result = {'account': account, 'repository': repo, 'private': True,
              'revision': commit.oid, 'all_six_files_readback_verified': True,
              'files': manifest['files']}
    Path(receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(receipt).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = publish(args.account, args.repo, args.artifacts, args.receipt)
    print(json.dumps({k: result[k] for k in ['repository', 'private', 'revision', 'all_six_files_readback_verified']}))
