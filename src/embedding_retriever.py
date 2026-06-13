from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import joblib
    import numpy as np
except ImportError:  # Optional enhancement; TF-IDF remains available.
    joblib = None
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .retriever import normalize_chunk_fields


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingUnavailableError(RuntimeError):
    pass


class EmbeddingRetriever:
    backend = "sentence_transformers"

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, model: Any = None):
        self.model_name = model_name
        self.model = model
        self.chunks: list[dict] = []
        self.embeddings = None

    @classmethod
    def dependency_available(cls) -> bool:
        return SentenceTransformer is not None and np is not None and joblib is not None

    def _ensure_model(self):
        if self.model is not None:
            return self.model
        if not self.dependency_available():
            raise EmbeddingUnavailableError(
                "Embedding retrieval requires sentence-transformers and numpy. "
                "Install requirements.txt; TF-IDF retrieval is still available."
            )
        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise EmbeddingUnavailableError(
                    f"Could not load embedding model '{self.model_name}': {exc}. "
                    "Check model availability or use retrieval.mode=tfidf."
                ) from exc
        return self.model

    def build_index(self, chunks: list[dict]) -> "EmbeddingRetriever":
        if not chunks:
            raise ValueError("Cannot build an embedding index without chunks")
        if np is None:
            raise EmbeddingUnavailableError("Embedding retrieval requires numpy")
        self.chunks = [normalize_chunk_fields(chunk) for chunk in chunks]
        texts = [f"{chunk.get('product', '')} {chunk['title']} {chunk['text']}" for chunk in self.chunks]
        try:
            vectors = self._ensure_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except EmbeddingUnavailableError:
            raise
        except Exception as exc:
            raise EmbeddingUnavailableError(f"Embedding model failed to encode documents: {exc}") from exc
        self.embeddings = self._normalize(np.asarray(vectors, dtype=float))
        return self

    build = build_index

    @staticmethod
    def _normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0, code_only: bool = False) -> list[dict]:
        if self.embeddings is None or not self.chunks:
            raise RuntimeError("Embedding index is not built")
        try:
            query_vector = self._ensure_model().encode([query], convert_to_numpy=True, show_progress_bar=False)
        except Exception as exc:
            raise EmbeddingUnavailableError(f"Embedding model failed to encode query: {exc}") from exc
        query_vector = self._normalize(np.asarray(query_vector, dtype=float))[0]
        scores = self.embeddings @ query_vector
        results = []
        for index in np.argsort(scores)[::-1]:
            chunk = normalize_chunk_fields(self.chunks[int(index)])
            if code_only and not chunk["has_code"]:
                continue
            score = float(scores[index])
            if score < min_score:
                continue
            results.append({**chunk, "score": round(score, 6)})
            if len(results) >= top_k:
                break
        return results

    def save(self, path: str | Path) -> None:
        if joblib is None:
            raise EmbeddingUnavailableError("Saving an embedding index requires joblib")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_name": self.model_name, "chunks": self.chunks, "embeddings": self.embeddings}, target)

    @classmethod
    def load(cls, path: str | Path, model: Any = None) -> "EmbeddingRetriever":
        if joblib is None:
            raise EmbeddingUnavailableError("Loading an embedding index requires joblib")
        payload = joblib.load(path)
        retriever = cls(payload["model_name"], model=model)
        retriever.chunks = [normalize_chunk_fields(chunk) for chunk in payload["chunks"]]
        retriever.embeddings = payload["embeddings"]
        return retriever
