from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    if yaml:
        with project_path(path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return {
        "paths": {"docs": "data/docs", "index": "data/index/tfidf_index.joblib", "output": "data/output"},
        "retrieval": {
            "mode": "tfidf",
            "top_k": 3,
            "min_score": 0.05,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "hybrid_alpha": 0.5,
        },
        "chunking": {"max_chars": 1200, "overlap": 120},
        "quality": {"min_citations": 1},
        "llm_answer": {
            "enabled": False,
            "mode": "template",
            "api_key_env": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_BASE_URL",
            "model_env": "OPENAI_MODEL",
        },
        "external_docs": {
            "enabled": True,
            "external_dir": "external",
            "imported_docs_dir": "data/docs_imported",
            "max_external_docs_per_source": 30,
            "max_file_size_kb": 512,
        },
    }


def ensure_dir(path: str | Path) -> Path:
    result = project_path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
