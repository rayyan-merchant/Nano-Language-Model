"""Report formatting must preserve actual model strings and evidence literals."""
from scripts.final.report import readable_prose


def test_prose_formatting_preserves_fenced_generations_hashes_and_links():
    checksum = 'a7' * 32
    text = ('F01 completed10000updates, all40 validations atstep250.\n'
            'https://example.org/F01_step10000 ' + checksum + ' `step_010000.pt`\nrawSHA256' + checksum + '\n'
            '[decision](F01_architecture_decision.md) step5000_generation.json\n'
            '```text\nROMEO:\ncompleted10000updates all40 malformedwords\n```\n')
    result = readable_prose(text)
    assert 'F01 completed 10000 updates, all 40 validations at step 250.' in result
    assert 'https://example.org/F01_step10000 ' + checksum + ' `step_010000.pt`' in result
    assert result.count(checksum) == 2
    assert '[decision](F01_architecture_decision.md) step5000_generation.json' in result
    assert '```text\nROMEO:\ncompleted10000updates all40 malformedwords\n```' in result
