import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.embedding_retriever import EmbeddingRetriever, EmbeddingUnavailableError


class FakeSentenceModel:
    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([
                float("rate" in lowered or "openai" in lowered),
                float("cuda" in lowered or "pytorch" in lowered),
                float("tokenizer" in lowered),
            ])
        return np.asarray(vectors, dtype=float)


class EmbeddingRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"content": "OpenAI rate limit backoff", "source": "rate.md", "product": "openai", "chunk_id": "chunk_001"},
            {"content": "PyTorch CUDA memory", "source": "cuda.md", "product": "pytorch", "chunk_id": "chunk_001"},
        ]

    def test_build_search_and_result_fields(self):
        result = EmbeddingRetriever(model=FakeSentenceModel()).build_index(self.chunks).search("OpenAI rate handling", top_k=1)[0]
        self.assertEqual(result["product"], "openai")
        for field in ("score", "source", "title", "section_title", "chunk_id", "source_path", "text", "has_code"):
            self.assertIn(field, result)

    def test_save_and_load(self):
        retriever = EmbeddingRetriever(model=FakeSentenceModel()).build_index(self.chunks)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "embedding.joblib"
            retriever.save(path)
            loaded = EmbeddingRetriever.load(path, model=FakeSentenceModel())
            self.assertEqual(loaded.search("CUDA", top_k=1)[0]["product"], "pytorch")

    def test_missing_dependency_has_clear_error(self):
        retriever = EmbeddingRetriever()
        with patch.object(retriever, "dependency_available", return_value=False):
            with self.assertRaisesRegex(EmbeddingUnavailableError, "TF-IDF retrieval is still available"):
                retriever.build_index(self.chunks)


if __name__ == "__main__":
    unittest.main()
