from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".md", ".mdx", ".txt", ".ipynb"}
IMPORT_MANIFEST = "_import_manifest.json"


def _load_notebook(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        if cell.get("cell_type") == "code":
            parts.append(f"```python\n{source.rstrip()}\n```")
        else:
            parts.append(source.strip())
    return "\n\n".join(parts)


def load_document(path: str | Path, metadata: dict | None = None) -> dict:
    source = Path(path)
    metadata = metadata or {}
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {source.suffix}")
    content = _load_notebook(source) if source.suffix.lower() == ".ipynb" else source.read_text(encoding="utf-8")
    return {
        "content": content,
        "source": metadata.get("source_file", source.name),
        "path": str(source),
        "product": metadata.get("source", source.parent.name),
        "extension": source.suffix.lower(),
        "title": metadata.get("title", source.stem.replace("_", " ")),
        "original_path": metadata.get("original_path", str(source)),
        "imported_path": metadata.get("imported_path"),
        "file_type": metadata.get("file_type", source.suffix.lower()),
        "is_imported": bool(metadata),
    }


def load_documents(directory: str | Path) -> list[dict]:
    root = Path(directory)
    if not root.exists():
        return []
    manifest_path = root / IMPORT_MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        documents = []
        for metadata in manifest.get("documents", []):
            imported = Path(metadata["imported_path"])
            if not imported.is_absolute():
                imported = root / imported
            if imported.exists() and imported.suffix.lower() in SUPPORTED_EXTENSIONS:
                documents.append(load_document(imported, metadata=metadata))
        return documents
    files: Iterable[Path] = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return [load_document(path) for path in files]
