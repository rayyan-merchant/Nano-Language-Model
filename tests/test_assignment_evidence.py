"""Submission-specific numerical and provenance checks, with no research training."""
import csv
from pathlib import Path

import numpy as np
import pytest

from tools import assignment_evidence as evidence


def test_manual_reference_has_known_values_and_visibility():
    result = evidence.manual_attention()
    assert result['causal']['output'][0] == [1, 0]
    assert result['causal']['weights'][1][0] == pytest.approx(.3302384506733431)
    assert result['causal']['output'][2][0] == pytest.approx(.751744921742276)
    assert result['bidirectional']['weights'][0][2] > .4
    assert result['causal']['weights'][0][2] == 0
    np.testing.assert_allclose(np.sum(result['causal']['weights'], axis=1), 1)


def test_recorded_trained_evidence_is_consistent():
    audit, generations, request, response = evidence.audit_receipts()
    assert audit['future_attention_exactly_zero']
    assert len(audit['independent_executions']) == 2
    assert len(generations) == 30
    assert request.model_id == response.model_id == 'b01'
    assert response.checkpoint_step == 3000


def test_corrupted_probability_receipt_is_rejected(monkeypatch):
    original = evidence.read

    def corrupted(path):
        result = original(path)
        if path.endswith('attention/receipt.json'):
            result['attention']['probabilities'][0][1] = .01
        return result

    monkeypatch.setattr(evidence, 'read', corrupted)
    with pytest.raises(AssertionError):
        evidence.audit_receipts()


def test_three_real_configuration_counts_and_tensor_trace():
    traces = evidence.trace_models(evidence.read('docs/submission/models.json'))
    assert [traces[mid]['parameters'] for mid in ['e02','b01','f01']] == [2173056,3062784,6046560]
    assert traces['e02']['shapes']['head_split'] == [1,6,12,32]
    assert traces['f01']['shapes']['head_split'] == [1,6,12,40]
    assert traces['e02']['shapes']['logits'] == [1,12,1024]


def test_generated_inventory_matches_receipt_and_original_sources():
    root = Path(__file__).resolve().parents[1]
    audit = evidence.read('docs/submission/verification/assignment_audit.json')
    for name, expected in audit['generated_sha256'].items():
        assert evidence.sha(root / 'docs/submission/evidence' / name) == expected
    for path, expected in audit['inputs_sha256'].items():
        assert evidence.sha(root / path) == expected
    with (root / 'docs/submission/evidence/figure_09_ablations.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    assert [row['experiment'] for row in rows] == [f'E{i:02}' for i in range(1,15)]
    assert all(row['replicates'] == '1' and row['standard_deviation'] == '' for row in rows)


def test_every_compliance_row_has_an_exact_handout_anchor():
    root = Path(__file__).resolve().parents[1]
    text = (root / 'docs/submission/assignment_compliance_matrix.md').read_text(encoding='utf-8')
    ids = {line.split('|')[1].strip().split(' / ')[0] for line in text.splitlines()
           if line.startswith('|') and ' / ' in line.split('|')[1] and not line.startswith('| ID')}
    quotes = evidence.read('docs/submission/verification/handout_requirement_quotes.json')
    assert quotes['pages'] == 11
    assert ids == set(quotes['requirements']) and len(ids) == 103
    assert all(q['normalized_whitespace_substring_verified'] for q in quotes['requirements'].values())
