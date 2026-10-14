"""Independent ByteLevel reversal preserves multibyte boundaries and failures."""
from nano_lm.bpe import BPETokenizer
from scripts.final.reproduce_generation import generated_bytes


def test_reversible_byte_mapping_and_invalid_UTF8_are_preserved():
    tokenizer=BPETokenizer.fit('ROMEO:\nTo be é € Shakespeare\n'*5,300)
    for text in ['ROMEO:\n','é €','To be, or not to be']:
        assert generated_bytes(tokenizer,tokenizer.encode(text))==text.encode('utf-8')
    invalid=[i for i in range(tokenizer.vocab_size) if '\ufffd' in tokenizer.decode([i])]
    assert invalid
    raw=generated_bytes(tokenizer,[invalid[0]])
    try:raw.decode('utf-8')
    except UnicodeDecodeError:pass
    else:raise AssertionError('Expected incomplete ByteLevel byte to remain invalid')
