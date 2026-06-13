import json
import unittest
from unittest.mock import patch

from scripts.retrieval_eval import retrieval_eval
from src.embedding_retriever import EmbeddingUnavailableError
from src.utils import project_path


class RetrievalEvalTests(unittest.TestCase):
    def test_reports_generate_when_embedding_is_unavailable(self):
        with patch(
            "src.rag_pipeline.EmbeddingRetriever.build_index",
            side_effect=EmbeddingUnavailableError("test embedding unavailable"),
        ):
            report = retrieval_eval()
        self.assertEqual(set(report["modes"]), {"tfidf", "embedding", "hybrid"})
        self.assertEqual(report["modes"]["embedding"]["effective_mode"], "tfidf")
        json_path = project_path("data/output/retrieval_eval.json")
        md_path = project_path("data/output/retrieval_eval.md")
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertIn("hit_at_1", payload["modes"]["tfidf"])
        self.assertIn("Retrieval Mode Evaluation", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
