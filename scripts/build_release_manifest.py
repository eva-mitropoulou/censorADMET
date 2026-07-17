"""Build a checksum manifest for a frozen CensorADMET release or data archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = [
    "data/measurement_records.parquet",
    "data/endpoints",
    "data/fixed_splits",
    "data/provenance",
    "configs/manuscript-v1.0.0.json",
    "results/matrix_endpoint/all_results.parquet",
    "results/matrix_measurement/all_results.parquet",
    "results/matrix_admet/all_results.parquet",
    "results/synthesis",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "RELEASE_MANIFEST.json")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--release-tag", default="v1.0.0-paper")
    args = parser.parse_args()
    files = []
    for relative in DEFAULT_PATHS:
        path = ROOT / relative
        paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for item in paths:
            files.append({"path": str(item.relative_to(ROOT)), "bytes": item.stat().st_size, "sha256": sha256(item)})
    payload = {
        "release_version": args.version,
        "release_tag": args.release_tag,
        "git_commit_sha": git_sha(),
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "dataset_origins": {"measurement_and_aggregated_inputs": "ChEMBL release 36, DOI: 10.6019/CHEMBL.database.36"},
        "licences": {"software": "MIT", "chEMBL_derived_data_and_artifacts": "CC-BY-SA-3.0"},
        "validation": {
            "commands": ["make test", "make verify", "make validate-splits"],
            "representative_value_tolerance": 0.01
        },
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(files)} checksums to {args.output}")


if __name__ == "__main__":
    main()
