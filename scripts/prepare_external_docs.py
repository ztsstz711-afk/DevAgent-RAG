from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.document_loader import IMPORT_MANIFEST, SUPPORTED_EXTENSIONS
from src.utils import load_config, project_path


SUPPORTED_SOURCES = (
    "openai-cookbook",
    "langchain-docs",
    "pytorch-tutorials",
    "vllm-docs",
    "llamafactory",
)
SKIP_DIRECTORIES = {".git", "node_modules", "__pycache__", "images", "assets", "build", "dist", ".venv"}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _read_text(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _title(path: Path, text: str) -> str:
    if path.suffix.lower() == ".ipynb":
        try:
            notebook = json.loads(text)
            text = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
        except json.JSONDecodeError:
            return path.stem.replace("_", " ")
    heading = re.search(r"(?m)^#{1,6}\s+(.+?)\s*$", text)
    return heading.group(1).strip() if heading else path.stem.replace("_", " ")


def _source_files(root: Path, max_size_bytes: int):
    for current, directories, files in os.walk(root):
        directories[:] = sorted(name for name in directories if name not in SKIP_DIRECTORIES)
        current_path = Path(current)
        for name in sorted(files):
            path = current_path / name
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > max_size_bytes:
                    continue
            except OSError:
                continue
            yield path


def prepare_external_docs(
    external_dir: str | Path | None = None,
    imported_docs_dir: str | Path | None = None,
    max_external_docs_per_source: int | None = None,
    max_file_size_kb: int | None = None,
) -> dict:
    config = load_config().get("external_docs", {})
    external_root = project_path(external_dir or config.get("external_dir", "external"))
    imported_root = project_path(imported_docs_dir or config.get("imported_docs_dir", "data/docs_imported"))
    limit = int(max_external_docs_per_source or config.get("max_external_docs_per_source", 30))
    max_size_kb = int(max_file_size_kb or config.get("max_file_size_kb", 512))
    imported_root.mkdir(parents=True, exist_ok=True)
    documents = []
    counts = {}
    skipped = {"missing_sources": [], "binary_or_invalid": 0, "oversized_or_unsupported": 0}

    if not external_root.exists():
        skipped["missing_sources"] = list(SUPPORTED_SOURCES)
    for source_name in SUPPORTED_SOURCES:
        source_root = external_root / source_name
        if not source_root.exists():
            if source_name not in skipped["missing_sources"]:
                skipped["missing_sources"].append(source_name)
            continue
        imported_count = 0
        for path in _source_files(source_root, max_size_kb * 1024):
            if imported_count >= limit:
                break
            text = _read_text(path)
            if text is None:
                skipped["binary_or_invalid"] += 1
                continue
            relative = path.relative_to(source_root)
            target = imported_root / source_name / relative
            _atomic_write(target, text)
            documents.append({
                "source": source_name,
                "source_file": path.name,
                "original_path": str(path.resolve()),
                "imported_path": target.relative_to(imported_root).as_posix(),
                "file_type": path.suffix.lower(),
                "title": _title(path, text),
            })
            imported_count += 1
        counts[source_name] = imported_count

    manifest = {
        "external_dir": str(external_root),
        "imported_docs_dir": str(imported_root),
        "max_external_docs_per_source": limit,
        "max_file_size_kb": max_size_kb,
        "total_imported": len(documents),
        "documents_per_source": counts,
        "skipped": skipped,
        "documents": documents,
    }
    _atomic_write(imported_root / IMPORT_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    result = prepare_external_docs()
    print(f"Imported {result['total_imported']} external documents into {result['imported_docs_dir']}")
    for source, count in result["documents_per_source"].items():
        print(f"  {source}: {count}")
    if result["skipped"]["missing_sources"]:
        print("Missing external sources: " + ", ".join(result["skipped"]["missing_sources"]))
