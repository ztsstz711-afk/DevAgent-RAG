from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.doc_normalizer import normalize_document
from src.document_loader import load_documents
from src.retriever import TfidfRetriever
from src.text_splitter import split_documents
from src.utils import load_config, project_path, write_json


def build_index(
    config_path: str = "configs/default.yaml",
    docs_dir: str | Path | None = None,
    imported_docs_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    stats_path: str | Path | None = None,
) -> tuple[int, Path]:
    config = load_config(config_path)
    sample_root = project_path(docs_dir or config["paths"]["docs"])
    external_config = config.get("external_docs", {})
    imported_root = project_path(
        imported_docs_dir or external_config.get("imported_docs_dir", "data/docs_imported")
    )

    sample_documents = load_documents(sample_root)
    imported_documents = load_documents(imported_root) if external_config.get("enabled", True) else []
    imported_documents = [{**document, "is_imported": True} for document in imported_documents]
    documents = [normalize_document(document) for document in [*sample_documents, *imported_documents]]
    if not documents:
        raise ValueError("Cannot build an index without sample or imported documents")
    chunks = split_documents(documents, **config["chunking"])

    target = project_path(index_path or config["paths"]["index"])
    TfidfRetriever().build(chunks).save(target)

    documents_per_source = Counter(document["product"] for document in documents)
    chunks_per_source = Counter(chunk["product"] for chunk in chunks)
    stats = {
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "sources": sorted(documents_per_source),
        "documents_per_source": dict(sorted(documents_per_source.items())),
        "chunks_per_source": dict(sorted(chunks_per_source.items())),
        "imported_docs_count": sum(bool(document.get("is_imported")) for document in documents),
        "retrieval_mode": config.get("retrieval", {}).get("mode", "tfidf"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    stats_target = project_path(stats_path or Path(config["paths"]["output"]) / "index_stats.json")
    write_json(stats_target, stats)
    return len(chunks), target


if __name__ == "__main__":
    count, path = build_index()
    stats = json.loads(project_path("data/output/index_stats.json").read_text(encoding="utf-8"))
    print(f"Built TF-IDF index at {path}")
    print(f"total_documents: {stats['total_documents']}")
    print(f"total_chunks: {count}")
    print(f"sources: {', '.join(stats['sources'])}")
    print(f"chunks_per_source: {stats['chunks_per_source']}")
    print(f"imported_docs_count: {stats['imported_docs_count']}")
    print(f"Stats: {project_path('data/output/index_stats.json')}")
