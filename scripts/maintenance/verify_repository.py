"""Offline cleanup integrity, links, notebooks and CPU interface verification."""
import argparse
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

from scripts.maintenance.archive_refs import BASE, blobs, git, git_blob, ledger, relink


def files(root):
    tracked=git(root,'ls-files').decode().splitlines()
    new=git(root,'ls-files','--others','--exclude-standard').decode().splitlines()
    return sorted({n for n in tracked+new if (Path(root)/n).is_file()})


def markdown_links(root, names=None):
    root=Path(root).resolve();broken=[];external=[];checked=0
    for name in names or files(root):
        if not name.endswith('.md'):
            continue
        text=(root/name).read_text(encoding='utf-8')
        # Commands, generated samples and inline literals are not Markdown links.
        text=re.sub(r'(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$', '', text)
        text=re.sub(r'`[^`\n]*`','',text)
        targets=re.findall(r'!?\[[^\]\n]*\]\(([^)\n]+)\)',text)
        targets+=re.findall(r'(?m)^\s*\[[^\]]+\]:\s*(\S+)',text)
        for literal in targets:
            target=literal.strip()
            target=target[1:target.index('>')] if target.startswith('<') and '>' in target else target.split(' "')[0]
            if re.match(r'^[A-Za-z]:[/\\]',target):
                external.append({'source':name,'target':target,'reason':'External machine-local artifact reference'});continue
            url=urlsplit(target)
            if url.scheme or url.netloc:
                continue
            if not url.path:
                continue  # Heading-only anchors are not relocations.
            path=unquote(url.path)
            resolved=(root/path.lstrip('/')) if path.startswith('/') else (root/name).parent/path
            checked+=1
            if not resolved.exists():
                broken.append({'source':name,'target':target,'resolved':str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)})
    return {'checked_file_links':checked,'broken':broken,'external_local_references':external,
            'scope':'Relative file targets outside fenced/inline code; HTTP/private artifacts not downloaded; heading anchors not checked'}


def notebooks(root, names):
    records=[]
    for name in names:
        if not name.endswith('.ipynb'):
            continue
        value=json.loads((root/name).read_text(encoding='utf-8'))
        assert value['nbformat']==4 and isinstance(value['metadata'],dict) and isinstance(value['cells'],list),name
        compiled=magics=executed=0
        for cell in value['cells']:
            assert cell['cell_type'] in {'code','markdown','raw'} and isinstance(cell['metadata'],dict),name
            source=cell['source'];assert isinstance(source,(str,list))
            code=''.join(source) if isinstance(source,list) else source
            if cell['cell_type']=='code':
                assert isinstance(cell['outputs'],list) and (cell['execution_count'] is None or isinstance(cell['execution_count'],int))
                executed+=bool(cell['outputs']) or cell['execution_count'] is not None
                if re.search(r'(?m)^\s*[%!]',code):
                    magics+=1
                else:
                    compile(code,name,'exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT);compiled+=1
        records.append({'path':name,'cells':len(value['cells']),'executed_cells':executed,
                        'compiled_Python_cells':compiled,'IPython_magic_cells_structurally_checked':magics})
    return records


def link_regressions(root, result):
    # Current public tree has no grandfathered broken-link exceptions.
    return result['broken']


def documentation_updates(root):
    """Verify explicit reviewed documentation revisions without mutable exemptions."""
    root=Path(root).resolve()
    manifest=root/'docs/maintenance/document_updates.json'
    raw=manifest.read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(raw).hexdigest()=='3a98b3944ae49a91dc078118a84d04c3e039bfa6d8cc43b3c493e045cf5179e7', 'Documentation revision manifest changed'
    record=json.loads(raw)
    assert record['schema_version']==1
    for name,expected in record['files'].items():
        path=(root/name).resolve()
        assert path.is_relative_to(root/'docs/submission'),name
        assert path.suffix in {'.md','.json'},name
        assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected,name
    return record['files']


def integrity(root, baseline=None):
    root=Path(root);rows=ledger(root)
    doc_updates=documentation_updates(root)
    names=[r['path'] for r in rows]
    values=blobs(root,BASE,names)
    historical_checked=current_checked=link_only_checked=0
    protected_prefixes=('src/','configs/','experiments/configs/','assets/tokenizers/',
                        'report/','docs/submission/','notebooks/')
    mutable={'report/README.md','notebooks/README.md','docs/submission/README.md'}
    for row in rows:
        name=row['path'];old=values[name];path=root/name
        assert git_blob(old)==row['git_blob'],f'Historical blob changed: {name}'
        historical_checked+=1
        if row['action']=='ARCHIVE':
            assert not path.exists(),f'Archived path remains active: {name}'
            continue
        assert path.is_file(),f'Missing retained file: {name}'
        if not name.startswith(protected_prefixes) or name in mutable:
            continue
        actual=path.read_bytes()
        if name in doc_updates:
            # Exact reviewed text revision was checked independently above.
            pass
        elif name.endswith('.md'):
            expected=relink(old.decode('utf-8'),name,rows)
            assert actual.decode('utf-8').replace('\r\n','\n')==expected,name
            link_only_checked+=old.decode('utf-8')!=expected
        elif name.endswith(('.py','.json','.csv','.txt','.ipynb','.jsonl','.gitattributes')) and not name.startswith('assets/tokenizers/'):
            assert actual.replace(b'\r\n',b'\n')==old.replace(b'\r\n',b'\n'),name
        else:
            assert actual==old,name
        current_checked+=1
    from nano_lm.artifacts import digest
    core={n:(root/n).read_text(encoding='utf-8') for n in names if n.startswith('src/') and n.endswith('.py')}
    assert digest(core)=='d5b205aecec5913cd11d921c0a84899badd95a014a490a7beaab3d2598c940f8'
    aggregate={n:(root/n).read_text(encoding='utf-8') for n in names if n.startswith(('src/','configs/','experiments/configs/')) and Path(n).suffix in {'.py','.json'}}
    assert digest(aggregate)=='4dd230985194865487254e2e750b07f79aeabec491e4fa8f8e483fedda1de741'
    previous='e4f768d53038662f0f24bcb9cb84a5210ad38a85'
    old_names=[n for n in git(root,'ls-tree','-r','--name-only',previous).decode().splitlines() if n.startswith(('src/','configs/','experiments/configs/')) and Path(n).suffix in {'.py','.json'}]
    historical={n:raw.decode('utf-8').replace('\r\n','\n') for n,raw in blobs(root,previous,old_names).items()}
    assert digest(historical)=='cdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6'
    return {'canonical_paths_and_bytes_checked':current_checked,'historical_git_blobs_checked':historical_checked,
            'archived_paths':sum(r['action']=='ARCHIVE' for r in rows),'reviewed_documentation_updates':len(doc_updates),'link_only_research_migrations':link_only_checked,
            'science_fingerprint':digest(core),'current_aggregate':digest(aggregate),'historical_aggregate':digest(historical),
            'scope':'Current retained scientific/research/submission artifacts plus separate full historical Git-object verification; no blanket deletion exemptions'}


def interfaces(root):
    import torch
    from nano_lm.bpe import from_artifact
    from nano_lm.config import load_experiment
    from nano_lm.model import GPT
    torch.set_num_threads(1)
    imported=[]
    for path in sorted((root/'src/nano_lm').glob('*.py')):
        module='nano_lm' if path.stem=='__init__' else 'nano_lm.'+path.stem
        importlib.import_module(module);imported.append(module)
    tokenizers={}
    for kind,checksum in [('character','ed96787d2f3f9741d69eba15160f31634ba6755b47e4c261ffedbb975fa0af77'),
                          ('bpe','2d7b1e84c8055eb5b9483a3fe6e4e3c9e348f5a28506ba19a0dc1644f0b36d55')]:
        tok=from_artifact(json.loads((root/f'assets/tokenizers/{kind}_canonical.json').read_text(encoding='utf-8')))
        assert tok.checksum==checksum and tok.decode(tok.encode('To be, or not to be'))=='To be, or not to be'
        tokenizers[kind]=checksum
    expected={**dict.fromkeys(['E01','E03','E05','E06','E07','E11','E12'],1804800),
              'E02':2173056,'E04':1829376,'E08':2694528,'E09':3584256,'E10':1213440,
              'E13':1021824,'E14':2808960,'F01':6046560,'B01':3062784}
    configurations=[]
    for eid,count in sorted(expected.items()):
        variant=json.loads((root/f'experiments/configs/{eid}.json').read_text(encoding='utf-8'))
        config,_,_=load_experiment(root,eid,1024 if variant['tokenizer']=='bpe' else 65)
        model=GPT(config).eval();assert model.parameter_count()==count
        with torch.no_grad():output,_,_=model(torch.zeros((1,8),dtype=torch.long))
        assert torch.isfinite(output).all() and tuple(output.shape)==(1,8,config.vocab_size)
        configurations.append({'id':eid,'parameters':count,'context':config.context_length,'output_shape':list(output.shape)})
    from nano_lm.evaluate import evaluate
    assert callable(evaluate)
    commands=['prepare','train','generate','storage','results','experiments','verification','smoke','plot_run']
    for name in commands:
        result=subprocess.run([sys.executable,'-m','nano_lm.'+name,'--help'],cwd=root,capture_output=True,text=True)
        assert result.returncode==0 and 'usage:' in result.stdout.lower(),(name,result.stderr)
    return {'imports':imported,'tokenizers':tokenizers,'configurations':configurations,'CLI_help':commands,
            'evaluation_Python_API':str(inspect.signature(evaluate)),
            'scope':'CPU finite forwards/tokenizer roundtrip/help only; zero optimizer updates; no dataset/test likelihood/archive download'}


def verify(root, baseline, output):
    root=Path(root).resolve();names=files(root)
    report={'integrity':integrity(root,baseline),'links':markdown_links(root,names),'notebooks':notebooks(root,names),
            'interfaces':interfaces(root)}
    report['links']['new_broken_links']=link_regressions(root,report['links'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'protected':report['integrity']['canonical_paths_and_bytes_checked'],
                      'configurations':len(report['interfaces']['configurations']),'notebooks':len(report['notebooks']),
                      'broken_links':len(report['links']['broken']),'output':str(output)}))
    if report['links']['new_broken_links']:
        raise SystemExit('New broken relative file links; inspect written receipt before completion')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--baseline',type=Path,default=Path('docs/maintenance/public_tree.csv'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(args.root,args.baseline,args.output)
