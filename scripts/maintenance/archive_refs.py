"""Pinned public history and deterministic link-only migration of research prose."""
import csv
import hashlib
import io
from functools import lru_cache
from pathlib import Path
import posixpath
import re
import subprocess
from urllib.parse import unquote, urlsplit

BASE = '8f88edee89dfceb6d0a8c84e385a886c4e058356'
ORIGINAL_BASE = 'cc7e345cee92c122cb1f7ad56a0d7627636934db'
REPOSITORY = 'https://github.com/rayyan-merchant/Nano-Language-Model'


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def ledger(root):
    path=Path(root)/'docs/maintenance/public_tree.csv'
    assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()=='2b60ee623bbf449054afbda590a7f5f2e1cbc129249361e191225eaa6c05d308', 'Preservation ledger changed; review archive contract before updating'
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def blobs(root, base, names):
    """Read exact Git objects in one bounded-lifetime process, including missing checks."""
    query=''.join(f'{base}:{n}\n' for n in names).encode()
    data=subprocess.run(['git','cat-file','--batch'],cwd=root,input=query,capture_output=True,check=True).stdout
    stream=io.BytesIO(data); values={}
    for name in names:
        header=stream.readline().split()
        if len(header)!=3 or header[1]!=b'blob':
            raise ValueError(f'Missing historical object: {base}:{name}; fetch full Git history')
        values[name]=stream.read(int(header[2]))
        assert stream.read(1)==b'\n'
    return values


def relink(text, source, rows, base=BASE):
    """Only replace file/directory Markdown targets archived by this cleanup.

    Prose, numbers, captions and generated text are otherwise byte-preserved.
    Historical blob URLs keep originals available without hundreds of stubs.
    """
    removed={r['path'] for r in rows if r['action']=='ARCHIVE'}
    kept={r['path'] for r in rows if r['action']=='KEEP'}
    dirs={p for n in removed for p in list(Path(n).parents) if str(p)!='.'}
    gone_dirs={p.as_posix() for p in dirs if not any(n.startswith(p.as_posix()+'/') for n in kept)}
    def replace(match):
        target=match[2]
        if target.startswith('<'):
            return match[0]  # No migration candidates use this syntax in the baseline.
        url=urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            return match[0]
        name=posixpath.normpath(posixpath.join(posixpath.dirname(source),unquote(url.path)))
        if name not in removed and name not in gone_dirs:
            return match[0]
        is_image=match[1].startswith('!')
        prefix=f'https://raw.githubusercontent.com/rayyan-merchant/Nano-Language-Model/{base}/' if is_image else f'{REPOSITORY}/{"tree" if name in gone_dirs else "blob"}/{base}/'
        suffix=('#'+url.fragment) if url.fragment else ''
        return match[1]+prefix+name+suffix+match[3]
    return re.sub(r'(!?\[[^\]\n]*\]\()([^\)\n]+)(\))',replace,text)


INTERNAL_BASE = '5f3bfd8b3a95c346be9f70aa0404e1cdf2fbd430'


def internal_archive(root):
    """Explicit second cleanup scope; the original preservation ledger stays intact."""
    path=Path(root)/'docs/maintenance/internal_archive.csv'
    raw=path.read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()=='7f8b5fbad5d2180518d0f60f1d9973403caabd5a4205e3c3083c15df191da9f1', 'Internal archive ledger changed'
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    names=[r['path'] for r in rows]
    assert len(names)==len(set(names))
    for name in names:
        assert name.startswith(('report/','docs/','scripts/pilot/write_')),name
        assert '..' not in Path(name).parts,name
    return rows


def internal_links(root, text, source):
    rows=internal_archive(root)
    removed={r['path'] for r in rows}
    tracked=_internal_source_paths(str(Path(root).resolve()))
    scope=[dict(r,action='ARCHIVE') for r in rows]
    scope += [{'path':n,'action':'KEEP'} for n in tracked if n not in removed]
    return relink(text,source,scope,base=INTERNAL_BASE)


@lru_cache(maxsize=4)
def _internal_source_paths(root):
    # The source tree is an immutable commit, unlike a working index.
    return git(root,'ls-tree','-r','--name-only',INTERNAL_BASE).decode().splitlines()


def verify_internal_archive(root):
    rows=internal_archive(root)
    originals=blobs(root,INTERNAL_BASE,[r['path'] for r in rows])
    tracked=set(git(root,'ls-files').decode().splitlines())
    for row in rows:
        name=row['path']
        assert name not in tracked,f'Internal file still tracked: {name}'
        assert git_blob(originals[name])==row['git_blob'],name
        assert len(originals[name])==int(row['bytes']),name
    return {r['path'] for r in rows}


def verify_retained_history(root, prefix='report/'):
    """Verify retained evidence exactly, permitting only ledger-defined link targets."""
    rows=ledger(root)
    selected=[r for r in rows if r['path'].startswith(prefix)]
    originals=blobs(root,BASE,[r['path'] for r in selected])
    internal=verify_internal_archive(root)
    for row in selected:
        name=row['path']; path=Path(root)/name; old=originals[name]
        assert git_blob(old)==row['git_blob'],name
        if name in internal:
            continue
        if row['action']=='ARCHIVE':
            assert not path.exists(),f'Archived file still in public tree: {name}'
            continue
        if name in {'report/README.md'}:
            continue
        actual=path.read_bytes()
        if name.endswith('.md'):
            assert actual.replace(b'\r\n',b'\n').decode()==internal_links(root,relink(old.decode(),name,rows),name),name
        else:
            assert actual.replace(b'\r\n',b'\n')==old.replace(b'\r\n',b'\n') if name.endswith(('.json','.csv','.txt','.jsonl','.gitattributes')) else actual==old,name
    return len(selected)


def git_blob(data):
    import hashlib
    return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
