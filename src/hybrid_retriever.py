from __future__ import annotations

from .retriever import normalize_chunk_fields


class HybridRetriever:
    backend = "hybrid_tfidf_embedding"

    def __init__(self, tfidf_retriever, embedding_retriever, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("hybrid alpha must be between 0 and 1")
        self.tfidf_retriever = tfidf_retriever
        self.embedding_retriever = embedding_retriever
        self.alpha = alpha
        self.chunks = getattr(tfidf_retriever, "chunks", [])

    @staticmethod
    def _normalize(results: list[dict]) -> dict[tuple, float]:
        if not results:
            return {}
        scores = [float(item["score"]) for item in results]
        low, high = min(scores), max(scores)
        return {
            HybridRetriever._key(item): (1.0 if high == low else (float(item["score"]) - low) / (high - low))
            for item in results
        }

    @staticmethod
    def _key(item: dict) -> tuple:
        return (item.get("source_path") or item.get("path") or item.get("source"), item["chunk_id"])

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0, code_only: bool = False) -> list[dict]:
        candidate_k = max(top_k * 3, top_k)
        tfidf = self.tfidf_retriever.search(query, top_k=candidate_k, min_score=0.0, code_only=code_only)
        embedding = self.embedding_retriever.search(query, top_k=candidate_k, min_score=-1.0, code_only=code_only)
        tfidf_scores = self._normalize(tfidf)
        embedding_scores = self._normalize(embedding)
        merged = {self._key(item): normalize_chunk_fields(item) for item in [*tfidf, *embedding]}
        results = []
        for key, item in merged.items():
            final = self.alpha * embedding_scores.get(key, 0.0) + (1 - self.alpha) * tfidf_scores.get(key, 0.0)
            if final >= min_score:
                results.append({**item, "score": round(final, 6)})
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]
