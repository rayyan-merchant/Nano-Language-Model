"""Training-text-only deterministic character vocabulary, no normalization."""
import hashlib
import json
from pathlib import Path


class CharacterTokenizer:
    def __init__(self, characters):
        self.characters = list(characters)
        if self.characters != sorted(set(self.characters)):
            raise ValueError("Vocabulary must be unique and sorted")
        if not self.characters or any(len(c) != 1 for c in self.characters):
            raise ValueError("Vocabulary must contain characters")
        self.lookup = {c: i for i, c in enumerate(self.characters)}

    @classmethod
    def fit(cls, training_text: str):
        return cls(sorted(set(training_text)))

    def encode(self, text: str):
        try:
            return [self.lookup[c] for c in text]
        except KeyError as exc:
            raise ValueError(f"Unseen character {exc.args[0]!r}; no silent replacement") from exc

    def decode(self, ids):
        ids = list(ids)
        if any(i < 0 or i >= len(self.characters) for i in ids):
            raise ValueError("Token outside vocabulary")
        return "".join(self.characters[i] for i in ids)

    def artifact(self):
        return {"kind": "character", "characters": self.characters,
                "special_tokens": [], "normalization": "none"}

    @property
    def checksum(self):
        return hashlib.sha256(json.dumps(self.artifact(), ensure_ascii=False,
                              sort_keys=True).encode("utf-8")).hexdigest()

    def save(self, path: Path):
        path.write_text(json.dumps(self.artifact(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        return cls(json.loads(path.read_text(encoding="utf-8"))["characters"])
