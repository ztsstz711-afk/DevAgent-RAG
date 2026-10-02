"""Verify that v7 real-source snapshots are unchanged and indexed locally."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "output" / "real_source_manifest_v7.json"
IMPORT_ROOT = ROOT / "data" / "docs_imported"
IMPORT_MANIFEST_PATH = IMPORT_ROOT / "_import_manifest.json"
CASES_PATH = ROOT / "data" / "evals" / "real_source_cases_v7.json"
OFFICIAL_HOSTS = {
    "developers.openai.com",
    "docs.pytorch.org",
    "docs.langchain.com",
    "huggingface.co",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    indexed = json.loads(IMPORT_MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8")).get("cases", [])
    indexed_by_path = {
        str(document.get("imported_path", "")).replace("\\", "/"): document
        for document in indexed.get("documents", [])
    }
    details = []
    for entry in manifest.get("entries", []):
        relative_output = Path(entry["output_file"])
        path = ROOT / relative_output
        indexed_path = str(path.relative_to(IMPORT_ROOT)).replace("\\", "/")
        parsed = urlparse(entry["resolved_url"])
        frozen_cases = [
            case for case in cases
            if case.get("source_type") == "official_documentation"
            and case.get("source_url") == entry["resolved_url"]
            and case.get("expected_source") == path.name
        ]
        checks = {
            "file_exists": path.is_file(),
            "sha256_matches": path.is_file() and _sha256(path) == entry["sha256"],
            "official_host": parsed.scheme == "https" and parsed.hostname in OFFICIAL_HOSTS,
            "indexed": indexed_path in indexed_by_path,
            "indexed_source_matches": (
                indexed_path in indexed_by_path
                and indexed_by_path[indexed_path].get("source") == entry.get("product")
            ),
            "has_matching_frozen_case": bool(frozen_cases),
        }
        details.append({
            "source_id": entry["source_id"],
            "output_file": str(relative_output).replace("\\", "/"),
            "resolved_url": entry["resolved_url"],
            "checks": checks,
            "passed": all(checks.values()),
        })
    report = {
        "verification_type": "real_source_provenance_and_index_check",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "entry_count": len(details),
        "passed": bool(details) and all(item["passed"] for item in details),
        "details": details,
    }
    output = ROOT / "data" / "output" / "real_source_verification_v7.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = verify()
    print(f"Verified {report['entry_count']} source snapshots: {'PASS' if report['passed'] else 'FAIL'}")
    raise SystemExit(0 if report["passed"] else 1)
