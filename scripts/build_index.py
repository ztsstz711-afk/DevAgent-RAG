from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.doc_normalizer import normalize_document
from src.document_loader import load_documents
from src.retriever import TfidfRetriever
from src.text_splitter import split_documents
from src.utils import load_config, project_path


def build_index() -> tuple[int, Path]:
    config = load_config()
    documents = [normalize_document(document) for document in load_documents(project_path(config["paths"]["docs"]))]
    chunks = split_documents(documents, **config["chunking"])
    target = project_path(config["paths"]["index"])
    TfidfRetriever().build(chunks).save(target)
    return len(chunks), target


if __name__ == "__main__":
    count, path = build_index()
    print(f"Built TF-IDF index with {count} chunks at {path}")
