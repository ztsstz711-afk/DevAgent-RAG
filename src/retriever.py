from __future__ import annotations

from pathlib import Path

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    RETRIEVER_BACKEND = "sklearn_tfidf"
except ImportError:  # Standard-library fallback for offline bootstrap environments.
    joblib = None
    from .compat import SimpleTfidfVectorizer as TfidfVectorizer
    from .compat import load_pickle, simple_cosine as cosine_similarity, dump_pickle
    RETRIEVER_BACKEND = "fallback"


def normalize_chunk_fields(chunk: dict) -> dict:
    content = chunk.get("content", chunk.get("text", ""))
    source_path = chunk.get("source_path", chunk.get("path", ""))
    return {
        **chunk,
        "content": content,
        "text": content,
        "source_path": source_path,
        "path": source_path,
        "title": chunk.get("title", chunk.get("source", "").rsplit(".", 1)[0].replace("_", " ")),
        "section_title": chunk.get("section_title", ""),
        "has_code": chunk.get("has_code", "```" in content),
    }


class TfidfRetriever:
    def __init__(self, vectorizer=None, matrix=None, chunks=None):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks or []
        self.backend = RETRIEVER_BACKEND

    def build(self, chunks: list[dict]) -> "TfidfRetriever":
        if not chunks:
            raise ValueError("Cannot build an index without chunks")
        self.chunks = [normalize_chunk_fields(chunk) for chunk in chunks]
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")
        search_texts = (
            f"{chunk.get('product', '')} {chunk.get('source', '').replace('_', ' ')} {chunk['content']}"
            for chunk in chunks
        )
        self.matrix = self.vectorizer.fit_transform(search_texts)
        return self

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0, code_only: bool = False) -> list[dict]:
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Index is not built")
        query_vector = self.vectorizer.transform([query])
        raw_scores = cosine_similarity(query_vector, self.matrix)
        scores = raw_scores.ravel() if hasattr(raw_scores, "ravel") else raw_scores[0]
        ranked = scores.argsort()[::-1] if hasattr(scores, "argsort") else sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[dict] = []
        for index in ranked:
            chunk = normalize_chunk_fields(self.chunks[int(index)])
            if code_only and "```" not in chunk["content"]:
                continue
            if float(scores[index]) < min_score:
                continue
            results.append({**chunk, "score": round(float(scores[index]), 6)})
            if len(results) >= top_k:
                break
        return results

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"vectorizer": self.vectorizer, "matrix": self.matrix, "chunks": self.chunks}
        joblib.dump(payload, target) if joblib else dump_pickle(payload, target)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfRetriever":
        payload = joblib.load(path) if joblib else load_pickle(path)
        retriever = cls(**payload)
        retriever.chunks = [normalize_chunk_fields(chunk) for chunk in retriever.chunks]
        return retriever
