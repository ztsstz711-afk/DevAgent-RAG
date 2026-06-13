import unittest

from src.hybrid_retriever import HybridRetriever


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.chunks = results

    def search(self, _query, **_kwargs):
        return [dict(item) for item in self.results]


class HybridRetrieverTests(unittest.TestCase):
    def test_fuses_scores_and_deduplicates_chunks(self):
        shared = {"content": "shared", "source": "a.md", "path": "a.md", "product": "openai", "chunk_id": "chunk_001"}
        tfidf = FakeRetriever([
            {**shared, "score": 0.9},
            {"content": "tfidf", "source": "b.md", "path": "b.md", "product": "langchain", "chunk_id": "chunk_001", "score": 0.2},
        ])
        embedding = FakeRetriever([
            {**shared, "score": 0.8},
            {"content": "embedding", "source": "c.md", "path": "c.md", "product": "pytorch", "chunk_id": "chunk_001", "score": 0.7},
        ])
        results = HybridRetriever(tfidf, embedding, alpha=0.5).search("query", top_k=3)
        identities = {(item["source"], item["chunk_id"]) for item in results}
        self.assertEqual(len(results), 3)
        self.assertEqual(len(identities), 3)
        self.assertEqual(results[0]["source"], "a.md")

    def test_rejects_invalid_alpha(self):
        with self.assertRaises(ValueError):
            HybridRetriever(FakeRetriever([]), FakeRetriever([]), alpha=1.5)


if __name__ == "__main__":
    unittest.main()
