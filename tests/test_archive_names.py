import pytest
from scripts.pilot.archive_names import archive_name


def test_explicit_name_requires_no_legacy_run_id():
    assert archive_name({'archive_file':'B01_development.zip'})=='B01_development.zip'
    assert archive_name({'archive_file':'F01_seed42_final.zip','run_id':'unused'})=='F01_seed42_final.zip'
    assert archive_name({'run_id':'E14_seed42_uuid'})=='E14_seed42_pilot.zip'


@pytest.mark.parametrize('name',['../bad.zip','/bad.zip','D:bad.zip','a/b.zip','a\\b.zip',None,''])
def test_unsafe_manifest_names_fail(name):
    with pytest.raises(ValueError):archive_name({'archive_file':name})
