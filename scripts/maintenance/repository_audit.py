"""Offline tracked-file inventory, dependency review and preservation baseline."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import subprocess

BASE = 'cc7e345cee92c122cb1f7ad56a0d7627636934db'
MOVES = {
    'handwritten-assignment.pdf': 'docs/handwritten-assignment.pdf',
    'requirements-cpu-windows.lock.txt': 'docs/qualifications/requirements-cpu-windows.lock.txt',
}
REMOVALS = {
    'data/.gitkeep': 'data/README.md already retains this directory',
    'checkpoints/.gitkeep': 'No runtime requires this root placeholder; weights are external',
    'outputs/figures/.gitkeep': 'Writers create parents; local generated figures remain untouched',
    'outputs/logs/.gitkeep': 'Writers create parents; local journals remain untouched',
    'outputs/samples/.gitkeep': 'Writers create parents; local samples remain untouched',
    'experiments/analysis/README.md': 'Collector guidance consolidated in docs/results_analysis.md',
}
CONSOLIDATIONS = {
    **{f'scripts/pilot/verify_e{eid}_archive.py': 'Bounded checksum and snapshot helpers in archive_tools.py'
       for eid in ['02', '05', '07', '12']},
    'scripts/pilot/e12_context_audit.py': 'Reuse the identical public context_audit.token_budget helper',
}


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def blobs(root, base, names):
    requests=''.join(f'{base}:{name}\n' for name in names).encode()
    result=subprocess.run(['git','cat-file','--batch'],cwd=root,input=requests,capture_output=True,check=True)
    stream=io.BytesIO(result.stdout);values={}
    for name in names:
        header=stream.readline().decode().split()
        assert len(header)==3 and header[1]=='blob',name
        values[name]=stream.read(int(header[2]));assert stream.read(1)==b'\n'
    return values


def protected(path):
    if path in {'report/README.md', 'notebooks/README.md'}:
        return False
    return path.startswith(('src/', 'configs/', 'experiments/configs/', 'assets/tokenizers/',
                            'notebooks/', 'docs/evidence/', 'docs/qualifications/evidence/',
                            'docs/qualifications/failures/')) or (
        path.startswith('report/') and path != 'report/README.md')


def purpose(path, text):
    if path.endswith('.py'):
        tree = ast.parse(text)
        return (ast.get_docstring(tree) or 'Python checks/interfaces: ' + ', '.join(
            n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)))).split('\n')[0]
    if path.endswith('.md'):
        return next((line.lstrip('# ').strip() for line in text.splitlines() if line.startswith('#')), 'Documentation')
    if path.endswith('.ipynb'):
        nb = json.loads(text)
        executed = sum(bool(c.get('outputs')) or c.get('execution_count') is not None for c in nb['cells'])
        return f'Notebook: {len(nb["cells"])} cells; {executed} cells with execution/output evidence'
    if path.endswith('.json'):
        value = json.loads(text)
        return 'Structured artifact: ' + ', '.join(list(value)[:8]) if isinstance(value, dict) else 'Structured artifact list'
    if path.endswith(('.png', '.pdf', '.jpg')):
        return 'Handwritten assignment evidence' if 'handwritten' in path else 'Recorded figure/evidence: ' + Path(path).stem
    if path.endswith('.csv'):
        return 'Table: ' + text.splitlines()[0][:160]
    if path.endswith('.gitkeep'):
        return 'Empty directory placeholder'
    return 'Installation/automation/provenance file: ' + Path(path).name


def classify(path):
    if path == 'docs/evidence/verify_gpu_export.py':
        return 'UNCERTAIN'
    if path == 'scripts/pilot/archive_tools.py':
        return 'ACTIVE_UTILITY'
    if path in REMOVALS:
        return 'REDUNDANT'
    if path.startswith(('src/', 'configs/', 'experiments/configs/', 'assets/tokenizers/')):
        return 'CORE'
    if path.startswith('scripts/') or path == 'docs/evidence/verify_gpu_export.py':
        return 'HISTORICAL_UTILITY' if 'qualification' in path or path.startswith('scripts/pilot/') or '/evidence/' in path else 'ACTIVE_UTILITY'
    if path.startswith('tests/') or path.startswith('.github/') or path in {'pyproject.toml', 'requirements.txt', 'requirements-cpu-windows.lock.txt', 'handwritten-assignment.pdf'}:
        return 'SUBMISSION'
    if protected(path) or path.endswith('.provenance.json'):
        return 'RESEARCH_EVIDENCE'
    return 'DOCUMENTATION'


def refresh(root, output):
    """Update dispositions/current-file coverage without redefining baseline hashes."""
    root,output=Path(root).resolve(),Path(output)
    path=output/'repository_inventory.csv'
    with path.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    current=set(git(root,'ls-files').decode().splitlines()+git(root,'ls-files','--others','--exclude-standard').decode().splitlines())
    current={n for n in current if (root/n).is_file()}
    modified=set(git(root,'diff','--name-only',BASE,'--').decode().splitlines())
    represented=set()
    for row in rows:
        name=row['path'];destination=MOVES.get(name,name);represented.add(destination)
        row['classification']=classify(name)
        if name in MOVES:
            row.update(proposed_action='MOVE',destination=destination)
        elif name in REMOVALS:
            row.update(proposed_action='REMOVE',reason=REMOVALS[name])
        elif name in CONSOLIDATIONS:
            row.update(proposed_action='CONSOLIDATE',reason=CONSOLIDATIONS[name])
        elif not row['baseline_git_sha256']:
            row.update(proposed_action='CREATE')
        elif name in modified:
            assert not protected(name),name
            row.update(proposed_action='UPDATE',reason='Current navigation/status, CI or extended compatibility tests; historical bodies/values preserved where applicable')
        elif name=='docs/evidence/verify_gpu_export.py':
            row.update(reason='Fixed historical local paths; retain unique provenance for manual review',safety_assessment='Retain; not a portable supported entry point')
    for name in sorted(current-represented):
        try:text=(root/name).read_text(encoding='utf-8-sig')
        except UnicodeError:text=''
        references=[]
        for caller in sorted(current):
            if caller==name or not caller.endswith(('.py','.md','.ipynb','.yml')):
                continue
            source=(root/caller).read_text(encoding='utf-8-sig')
            if name in source:
                references.append(caller+' [current literal path]')
            elif name.endswith('.py') and caller.endswith('.py'):
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node,ast.ImportFrom) and node.module:
                        module=node.module.replace('.','/')+'.py'
                        if name in {module,(Path(caller).parent/module).as_posix()}:
                            references.append(caller+' [current Python import]');break
        rows.append(dict(path=name,classification=classify(name),proposed_action='CREATE',destination='',
                         purpose=purpose(name,text),reason='Purposeful cleanup navigation, shared helper, regression check or maintenance evidence',
                         references_dependencies='; '.join(references),provenance_relevance='Cleanup artifact; not research measurement',
                         safety_assessment='Offline/tested; source/evidence protection applies',baseline_git_sha256=''))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert current <= {r['destination'] or r['path'] for r in rows if r['proposed_action']!='REMOVE'}
    print(json.dumps({'current_files_covered':len(current),'inventory_rows_including_removed':len(rows)}))


def audit(root, output, base=BASE):
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = git(root, 'ls-tree', '-r', '--name-only', base).decode().splitlines()
    original=blobs(root,base,names)
    sources = {}
    for name in names:
        try:
            sources[name] = (root / name).read_text(encoding='utf-8-sig')
        except (UnicodeError, OSError):
            sources[name] = ''
    refs = defaultdict(set)
    # Search all text, including Markdown, CI, notebook cells/outputs and runtime literals.
    for caller, text in sources.items():
        for target in names:
            if caller != target and target in text:
                refs[target].add(caller + ' [literal path]')
        if caller.endswith('.py'):
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module.replace('.', '/') + '.py'
                    candidates = [module, str(Path(caller).parent / module).replace('\\', '/')]
                    if node.module.startswith('nano_lm.'):
                        candidates.append('src/' + module)
                    candidates.extend(node.module.replace('.', '/') + '/' + item.name + '.py' for item in node.names)
                    for candidate in candidates:
                        if candidate in names:
                            refs[candidate].add(caller + ' [Python import]')
                elif isinstance(node, ast.Import):
                    for item in node.names:
                        candidate=item.name.replace('.', '/') + '.py'
                        if candidate in names:
                            refs[candidate].add(caller + ' [Python import]')
    rows = []
    for name in names:
        action = 'MOVE' if name in MOVES else 'REMOVE' if name in REMOVALS else 'CONSOLIDATE' if name in CONSOLIDATIONS else 'KEEP'
        reason = REMOVALS.get(name) or CONSOLIDATIONS.get(name) or (
            'Relocate without byte changes; only historical plain-text paths found' if name in MOVES else
            'Stable scientific/provenance path and original bytes retained' if protected(name) else
            'Unique documented workflow or project guidance retained')
        rows.append(dict(path=name, classification=classify(name), proposed_action=action,
                         destination=MOVES.get(name, ''), purpose=purpose(name, sources[name]), reason=reason,
                         references_dependencies='; '.join(sorted(refs[name])),
                         provenance_relevance='Frozen scientific/artifact evidence' if protected(name) else 'Historical Git version retained',
                         safety_assessment='Preserve' if action == 'KEEP' else 'Dependency-audited; verify before/after',
                         baseline_git_sha256=hashlib.sha256(original[name]).hexdigest()))
    with (output / 'repository_inventory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    integrity = {}
    for name in names:
        if not protected(name):
            continue
        raw = original[name]
        working = (root / name).read_bytes()
        mode = 'exact' if working == raw else 'crlf_to_lf'
        assert working == raw or working.replace(b'\r\n', b'\n') == raw, name
        integrity[name] = {'git_sha256': hashlib.sha256(raw).hexdigest(),
                           'working_sha256': hashlib.sha256(working).hexdigest(), 'checkout_normalization': mode}
    data = {'starting_main': base, 'baseline_platform': platform.system(), 'tracked_files': len(names),
            'directory_counts': dict(sorted(Counter(n.split('/')[0] if '/' in n else '(root)' for n in names).items())),
            'scripts': sum(n.startswith('scripts/') for n in names),
            'notebook_files': sum(n.endswith('.ipynb') for n in names),
            'notebook_directory_files': sum(n.startswith('notebooks/') for n in names), 'protected': integrity}
    (output / 'baseline_integrity.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    reviews = []
    for name in names:
        if not name.endswith('.py'):
            continue
        tree = ast.parse(sources[name])
        reviews.append({'file': name, 'lines': len(sources[name].splitlines()),
                        'imports': [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))],
                        'interfaces': [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))],
                        'CLI_guard': any(isinstance(n, ast.If) and '__name__' in ast.unparse(n.test) for n in tree.body)})
    (output / 'python_dependency_review.json').write_text(json.dumps(reviews, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in data.items() if k != 'protected'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('docs/maintenance'))
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--refresh-current',action='store_true',help='Refresh current dispositions; preserve the original integrity baseline')
    args = parser.parse_args()
    if args.refresh_current:refresh(args.root,args.output)
    else:audit(args.root,args.output,args.base)
