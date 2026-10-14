"""Cleanup must preserve scientific bytes and historical utility contracts."""
import ast
import hashlib
import importlib
import io
from pathlib import Path
import subprocess
import sys

import pytest
from nano_lm.config import ModelConfig
from nano_lm.train import TrainSettings
from scripts.maintenance.repository_audit import BASE, git
from scripts.maintenance.verify_repository import markdown_links, link_regressions
from scripts.pilot.archive_tools import checksum, reconstruct_snapshots
from scripts.pilot.context_audit import token_budget

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('eid',['02','05','07','12'])
def test_archive_public_aliases_preserve_shared_helpers(eid):
    module=importlib.import_module(f'scripts.pilot.verify_e{eid}_archive')
    assert module.checksum is checksum
    if eid in {'07','12'}:
        assert module.reconstruct_snapshots is reconstruct_snapshots


def test_hash_is_bounded_and_exact_across_chunk_boundaries():
    content=b'first\x00'+bytes(range(256))*17000+b'last'
    reads=[]
    class Stream(io.BytesIO):
        def read(self,size=-1):
            reads.append(size);return super().read(size)
    class File:
        def open(self,mode):
            assert mode=='rb';return Stream(content)
    assert checksum(File())==hashlib.sha256(content).hexdigest()
    assert len(reads)>4 and set(reads)=={2**20}


@pytest.mark.parametrize('context,accumulation,windows',[(64,16,64),(128,8,32),(256,4,16)])
def test_context_helper_alias_keeps_exact_token_grouping(context,accumulation,windows):
    from scripts.pilot.e12_context_audit import token_budget as e12_budget
    assert e12_budget is token_budget
    result=e12_budget(ModelConfig(65,context_length=context),TrainSettings())
    assert result['accumulation']==accumulation and result['sampled_windows_per_update']==windows
    assert result['predicted_positions_per_update']==4096
    with pytest.raises(ValueError):e12_budget(ModelConfig(65,context_length=context),TrainSettings(tokens_per_update=4095))


@pytest.mark.parametrize('file,symbol',[
    *[(f'scripts/pilot/verify_e{eid}_archive.py','verify') for eid in ['02','05','07','12']],
    ('scripts/pilot/e12_context_audit.py','audit'),
])
def test_retained_archive_and_numerical_bodies_identical_to_reviewed_main(file,symbol):
    old=ast.parse(git(ROOT,'show',f'{BASE}:{file}').decode('utf-8'))
    current=ast.parse((ROOT/file).read_text(encoding='utf-8'))
    get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==symbol)
    assert ast.dump(get(old))==ast.dump(get(current))


@pytest.mark.parametrize('eid',['02','05','07','12'])
def test_direct_archive_script_help_remains_available(eid):
    result=subprocess.run([sys.executable,f'scripts/pilot/verify_e{eid}_archive.py','--help'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0 and 'usage:' in result.stdout.lower()


def test_ignore_generated_data_secrets_and_archives_but_not_evidence():
    for path in ['outputs/run/metadata.json','outputs/.gitkeep','data/local-copy/input.txt','checkpoints/model.pt',
                 'download.part','evidence.zip','.kaggle/kaggle.json','.env.local','dist/package.whl']:
        result=subprocess.run(['git','check-ignore','--no-index','-q',path],cwd=ROOT)
        assert result.returncode==0,path
    for path in ['data/README.md','docs/handwritten-assignment.pdf','report/balanced/figures/combined_loss.pdf',
                 'assets/tokenizers/bpe_canonical.json','notebooks/kaggle_e12_seed42_research_executed.ipynb']:
        result=subprocess.run(['git','check-ignore','--no-index','-q',path],cwd=ROOT)
        assert result.returncode==1,path


def test_relative_markdown_file_targets_have_no_new_defects():
    result=markdown_links(ROOT)
    assert not link_regressions(ROOT,result),result['broken']


def test_known_historical_defect_does_not_hide_new_broken_source():
    result={'broken':[{'source':'docs/new.md','target':'generation.md'}]}
    assert link_regressions(ROOT,result)==result['broken']
