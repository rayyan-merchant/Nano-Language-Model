"""Verified immutable export snapshots. Completion manifest is the commit marker."""
import argparse
import json
from pathlib import Path
from .artifacts import atomic_bytes, atomic_json, file_hash, digest


def verify_snapshot(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "export_manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported export schema")
    unsigned = {k: v for k, v in manifest.items() if k != "hash"}
    if digest(unsigned) != manifest["hash"]:
        raise ValueError("Export manifest corruption")
    for relative, checksum in manifest["files"].items():
        path = (directory / relative).resolve()
        if directory not in path.parents or file_hash(path) != checksum:
            raise ValueError(f"Corrupt or unsafe export file: {relative}")
    return manifest


def export_run(run_dir, destination, allow_active=False):
    run_dir, destination = Path(run_dir), Path(destination)
    if (run_dir / ".active.lock").exists() and not allow_active:
        raise ValueError("Run is active; export from the trainer checkpoint boundary")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    summaries = sorted(run_dir.glob("summary_*.json"))
    checkpoints = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    if not checkpoints:
        raise ValueError("No completed checkpoint to export")
    # Preserve the latest and best checkpoint, plus all metadata/logs/summaries.
    from .checkpoint import read_checkpoint
    payload, _ = read_checkpoint(checkpoints[-1])
    best = payload["training_state"]["best_checkpoint"]
    selected = {checkpoints[-1].name, best}
    files = [p for p in run_dir.rglob("*") if p.is_file()
             and not p.name.endswith((".tmp", ".lock")) and not p.name.startswith(".")
             and ("checkpoints" not in p.relative_to(run_dir).parts or p.name in selected)]
    checksums = {p.relative_to(run_dir).as_posix(): file_hash(p) for p in files}
    snapshot = destination / f"{metadata['run_id']}_step{payload['step']:06d}_{digest(checksums)[:12]}"
    if snapshot.exists():
        verify_snapshot(snapshot)
        return snapshot
    snapshot.mkdir(parents=True, exist_ok=False)
    for path in files:
        relative = path.relative_to(run_dir).as_posix()
        atomic_bytes(snapshot / relative, path.read_bytes())
        if file_hash(snapshot / relative) != checksums[relative]:
            raise ValueError("Source changed during export; partial snapshot retained")
    manifest = {"schema_version": 1, "run_id": metadata["run_id"], "step": payload["step"],
                "config_hash": metadata["config_hash"], "files": checksums}
    manifest["hash"] = digest(manifest)
    atomic_json(snapshot / "export_manifest.json", manifest)
    verify_snapshot(snapshot)
    return snapshot


def restore_snapshot(snapshot, output_root):
    snapshot = Path(snapshot)
    manifest = verify_snapshot(snapshot)
    destination = Path(output_root) / manifest["run_id"]
    destination.mkdir(parents=True, exist_ok=False)
    for relative in manifest["files"]:
        atomic_bytes(destination / relative, (snapshot / relative).read_bytes())
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("run_dir", type=Path)
    export.add_argument("destination", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("snapshot", type=Path)
    restore = sub.add_parser("restore")
    restore.add_argument("snapshot", type=Path)
    restore.add_argument("output_root", type=Path)
    args = p.parse_args()
    if args.command == "export":
        print(export_run(args.run_dir, args.destination))
    elif args.command == "restore":
        print(restore_snapshot(args.snapshot, args.output_root))
    else:
        print(json.dumps(verify_snapshot(args.snapshot), indent=2))


if __name__ == "__main__":
    main()
