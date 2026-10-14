"""Postprocessing fixture tests; E02 text is not represented as F01 research."""
import json
from pathlib import Path
from scripts.final.generation_diagnostics import describe

ROOT=Path(__file__).resolve().parents[1]


def test_byte_BPE_length_aware_descriptors_and_complete_grid(tmp_path):
    # Reuse genuine E02 text solely as a temporary parser fixture. No F01
    # measurement/report is generated from this fixture outside the test folder.
    fixture=json.loads((ROOT/'report/pilots/E02_seed42/generation.json').read_text(encoding='utf-8'))
    fixture['fixture_notice']='Historical E02 outputs used for parser tests; not F01 results'
    source=tmp_path/'fixture.json';source.write_text(json.dumps(fixture),encoding='utf-8')
    dataset=tmp_path/'data';dataset.mkdir();(dataset/'train.txt').write_text('ROMEO:\nTo be, or not to be\n',encoding='utf-8')
    out=tmp_path/'diagnostics.json';describe(ROOT,source,dataset,out)
    observed=json.loads(out.read_text(encoding='utf-8'))
    assert len(observed['rows'])==600 and len(observed['matched_settings'])==100
    BPE=[r for r in observed['rows'] if r['experiment']=='F01']
    assert any(r['continuation_characters']!=256 for r in BPE)
    assert any(r['replacement_characters']>0 for r in BPE)
    assert all(len(v)==6 for v in observed['matched_settings'].values())
