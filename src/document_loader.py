from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".md", ".mdx", ".txt", ".ipynb"}


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


def load_document(path: str | Path) -> dict:
    source = Path(path)
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {source.suffix}")
    content = _load_notebook(source) if source.suffix.lower() == ".ipynb" else source.read_text(encoding="utf-8")
    return {
        "content": content,
        "source": source.name,
        "path": str(source),
        "product": source.parent.name,
        "extension": source.suffix.lower(),
    }


def load_documents(directory: str | Path) -> list[dict]:
    root = Path(directory)
    files: Iterable[Path] = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return [load_document(path) for path in files]
