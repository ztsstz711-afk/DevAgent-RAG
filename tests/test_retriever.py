import tempfile
import unittest
from pathlib import Path

from src.retriever import RETRIEVER_BACKEND, TfidfRetriever


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"content": "OpenAI rate limits use exponential backoff", "source": "rate.md", "product": "openai", "chunk_id": "chunk_001"},
            {"content": "PyTorch CUDA memory reduce batch size", "source": "cuda.md", "product": "pytorch", "chunk_id": "chunk_001"},
            {"content": "```python\nprint('code')\n```", "source": "code.md", "product": "demo", "chunk_id": "chunk_001"},
        ]
        self.retriever = TfidfRetriever().build(self.chunks)

    def test_returns_relevant_result_first(self):
        result = self.retriever.search("OpenAI rate limit", top_k=1)
        self.assertEqual(result[0]["product"], "openai")

    def test_code_only(self):
        result = self.retriever.search("code", code_only=True)
        self.assertEqual(len(result), 1)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.joblib"
            self.retriever.save(path)
            loaded = TfidfRetriever.load(path)
            self.assertEqual(loaded.search("CUDA", top_k=1)[0]["product"], "pytorch")

    def test_backend_identifier_exists(self):
        self.assertIn(RETRIEVER_BACKEND, {"sklearn_tfidf", "fallback"})
        self.assertEqual(self.retriever.backend, RETRIEVER_BACKEND)


if __name__ == "__main__":
    unittest.main()
