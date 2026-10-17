"""Pinned public history and deterministic link-only migration of research prose."""
import csv
import io
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
    with (Path(root)/'docs/maintenance/public_tree.csv').open(encoding='utf-8', newline='') as f:
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


def relink(text, source, rows):
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
        prefix=f'https://raw.githubusercontent.com/rayyan-merchant/Nano-Language-Model/{BASE}/' if is_image else f'{REPOSITORY}/{"tree" if name in gone_dirs else "blob"}/{BASE}/'
        suffix=('#'+url.fragment) if url.fragment else ''
        return match[1]+prefix+name+suffix+match[3]
    return re.sub(r'(!?\[[^\]\n]*\]\()([^\)\n]+)(\))',replace,text)


def verify_retained_history(root, prefix='report/'):
    """Verify retained evidence exactly, permitting only ledger-defined link targets."""
    rows=ledger(root)
    selected=[r for r in rows if r['path'].startswith(prefix)]
    originals=blobs(root,BASE,[r['path'] for r in selected])
    for row in selected:
        name=row['path']; path=Path(root)/name; old=originals[name]
        assert git_blob(old)==row['git_blob'],name
        if row['action']=='ARCHIVE':
            assert not path.exists(),f'Archived file still in public tree: {name}'
            continue
        if name in {'report/README.md'}:
            continue
        actual=path.read_bytes()
        if name.endswith('.md'):
            assert actual.replace(b'\r\n',b'\n').decode()==relink(old.decode(),name,rows),name
        else:
            assert actual.replace(b'\r\n',b'\n')==old.replace(b'\r\n',b'\n') if name.endswith(('.json','.csv','.txt')) else actual==old,name
    return len(selected)


def git_blob(data):
    import hashlib
    return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
