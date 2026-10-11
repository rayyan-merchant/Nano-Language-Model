"""Byte-level BPE fitted exclusively on training text. No special tokens."""
import json
import hashlib
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from .tokenizer import CharacterTokenizer


class BPETokenizer:
    def __init__(self, backend):
        self.backend = backend

    @classmethod
    def fit(cls, training_text, vocab_size=1024):
        if vocab_size < 256:
            raise ValueError("Byte BPE needs at least 256 alphabet entries")
        backend = Tokenizer(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        backend.train_from_iterator([training_text], trainer=trainer)
        return cls(backend)

    @property
    def vocab_size(self):
        return self.backend.get_vocab_size()

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        if any(i < 0 or i >= self.vocab_size for i in ids):
            raise ValueError("Token outside vocabulary")
        return self.backend.decode(list(ids), skip_special_tokens=False)

    def byte_lengths(self, ids):
        # ByteLevel maps each input byte to one alphabet character, including
        # bytes inside multibyte UTF-8 characters. Do not use decoded token length.
        return [len(self.backend.id_to_token(i)) for i in ids]

    def artifact(self):
        return {"kind": "bpe", "backend": json.loads(self.backend.to_str()),
                "special_tokens": [], "normalization": "none"}

    @property
    def checksum(self):
        return hashlib.sha256(json.dumps(self.artifact(), ensure_ascii=False,
                              sort_keys=True).encode("utf-8")).hexdigest()


def from_artifact(artifact):
    if artifact["kind"] == "character":
        return CharacterTokenizer(artifact["characters"])
    if artifact["kind"] == "bpe":
        return BPETokenizer(Tokenizer.from_str(json.dumps(artifact["backend"])))
    raise ValueError("Unknown tokenizer kind")


def vocabulary_size(tokenizer):
    return len(tokenizer.characters) if isinstance(tokenizer, CharacterTokenizer) else tokenizer.vocab_size


def encode_with_bytes(tokenizer, text):
    ids = tokenizer.encode(text)
    if tokenizer.decode(ids) != text:
        raise ValueError("Tokenizer failed lossless reconstruction")
    lengths = ([len(c.encode("utf-8")) for c in text] if isinstance(tokenizer, CharacterTokenizer)
               else tokenizer.byte_lengths(ids))
    if len(ids) != len(lengths) or sum(lengths) != len(text.encode("utf-8")):
        raise ValueError("Tokenizer source-byte coverage mismatch")
    return ids, lengths
