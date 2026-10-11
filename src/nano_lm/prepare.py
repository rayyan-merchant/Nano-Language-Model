"""Pinned Tiny Shakespeare acquisition and immutable partition/tokenizer manifest."""
import argparse
import hashlib
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from .artifacts import atomic_bytes, atomic_json, digest, file_hash
from .bpe import BPETokenizer, from_artifact, encode_with_bytes, vocabulary_size
from .data import split_text
from .tokenizer import CharacterTokenizer

REVISION = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
EXPECTED_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
REPOSITORY = "https://github.com/karpathy/char-rnn.git"
SOURCE_PATH = "data/tinyshakespeare/input.txt"
URL = f"https://raw.githubusercontent.com/karpathy/char-rnn/{REVISION}/{SOURCE_PATH}"


def acquire(transport="http"):
    if transport == "http":
        with urllib.request.urlopen(URL, timeout=60) as response:
            return response.read()
    if transport != "git":
        raise ValueError("Transport must be http or git")
    with tempfile.TemporaryDirectory() as tmp:
        for command in (["git", "init", tmp], ["git", "-C", tmp, "remote", "add", "origin", REPOSITORY],
                        ["git", "-C", tmp, "fetch", "--depth=1", "origin", REVISION]):
            subprocess.run(command, check=True, capture_output=True)
        return subprocess.check_output(["git", "-C", tmp, "show", f"{REVISION}:{SOURCE_PATH}"])


def prepare(directory, raw=None, transport="http", bpe_vocab_size=1024):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        source_kind = "tiny_shakespeare" if raw is None else "test_fixture"
        raw = acquire(transport) if raw is None else raw
        if source_kind == "tiny_shakespeare" and hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
            raise ValueError("Pinned source checksum mismatch")
        text = raw.decode("utf-8", errors="strict")
        if text.encode("utf-8") != raw:
            raise ValueError("Source must be lossless UTF-8")
        partitions, split = split_text(text)
        atomic_bytes(directory / "source.txt", raw)
        manifest = {"schema_version": 1, "source": {"kind": source_kind, "repository": REPOSITORY, "revision": REVISION,
                    "path": SOURCE_PATH, "url": URL, "transport": transport,
                    "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
                    "split": split, "partitions": {}, "tokenizers": {}}
        for name, part in partitions.items():
            atomic_bytes(directory / f"{name}.txt", part.encode())
            manifest["partitions"][name] = {"bytes": len(part.encode()), "characters": len(part),
                                         "sha256": file_hash(directory / f"{name}.txt")}
        for kind, tokenizer in [("character", CharacterTokenizer.fit(partitions["train"])),
                                ("bpe", BPETokenizer.fit(partitions["train"], bpe_vocab_size))]:
            artifact = tokenizer.artifact()
            atomic_json(directory / f"{kind}.json", artifact)
            loaded = from_artifact(artifact)
            assert loaded.checksum == tokenizer.checksum
            counts = {name: len(encode_with_bytes(loaded, part)[0]) for name, part in partitions.items()}
            manifest["tokenizers"][kind] = {"hash": tokenizer.checksum, "vocab_size": vocabulary_size(tokenizer),
                "fit_partition": "train", "fit_sha256": manifest["partitions"]["train"]["sha256"],
                "tokens": counts}
        manifest["manifest_hash"] = digest(manifest)
        atomic_json(directory / "manifest.json", manifest)
        return validate(directory)
    except Exception:
        # Retain partial directory for diagnosis; never silently recycle it.
        raise


def validate(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if digest(unsigned) != manifest["manifest_hash"]:
        raise ValueError("Dataset manifest checksum mismatch")
    source = (directory / "source.txt").read_bytes()
    if file_hash(directory / "source.txt") != manifest["source"]["sha256"]:
        raise ValueError("Source corruption")
    if manifest["source"].get("kind") == "tiny_shakespeare" and file_hash(directory / "source.txt") != EXPECTED_SHA256:
        raise ValueError("Unexpected Tiny Shakespeare source")
    parts, split = split_text(source.decode("utf-8"))
    if split != manifest["split"]:
        raise ValueError("Split boundaries changed")
    for name, text in parts.items():
        if (directory / f"{name}.txt").read_bytes() != text.encode():
            raise ValueError("Partition mismatch or boundary leakage")
        if file_hash(directory / f"{name}.txt") != manifest["partitions"][name]["sha256"]:
            raise ValueError("Partition hash mismatch")
    for kind, info in manifest["tokenizers"].items():
        tokenizer = from_artifact(json.loads((directory / f"{kind}.json").read_text(encoding="utf-8")))
        if info["fit_partition"] != "train" or info["fit_sha256"] != manifest["partitions"]["train"]["sha256"]:
            raise ValueError("Tokenizer provenance mismatch")
        if tokenizer.checksum != info["hash"] or vocabulary_size(tokenizer) != info["vocab_size"]:
            raise ValueError("Tokenizer hash/vocabulary mismatch")
        for name, text in parts.items():
            if len(encode_with_bytes(tokenizer, text)[0]) != info["tokens"][name]:
                raise ValueError("Token count mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=["http", "git"], default="http")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    result = validate(args.output) if args.validate else prepare(args.output, transport=args.transport)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
