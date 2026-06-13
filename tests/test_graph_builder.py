import unittest

from src.graph_builder import GRAPH_BACKEND, build_graph
from src.retriever import TfidfRetriever


class GraphBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chunks = [
            {"content": "CUDA out of memory: reduce batch size.\n```python\nimport torch\ntorch.cuda.empty_cache()\n```", "source": "cuda.md", "product": "pytorch", "chunk_id": "chunk_001"},
            {"content": "OpenAI rate limits require exponential backoff and jitter.", "source": "rate.md", "product": "openai", "chunk_id": "chunk_001"},
        ]
        cls.graph = build_graph(TfidfRetriever().build(chunks), {"retrieval": {"top_k": 2, "min_score": 0.0}})

    def test_question_route(self):
        result = self.graph.invoke({"question": "OpenAI rate limits", "tool_trace": []})
        self.assertEqual(result["route"], "question")
        self.assertTrue(result["quality"]["passed"])
        self.assertEqual(result["tool_trace"][1]["tool"], "doc_search_tool")

    def test_error_route(self):
        result = self.graph.invoke({"question": "CUDA out of memory", "tool_trace": []})
        self.assertEqual(result["route"], "error")
        self.assertEqual(result["error_info"]["primary_type"], "cuda_oom")
        self.assertEqual(result["tool_trace"][1]["tool"], "error_parser_tool")

    def test_backend_identifier_exists(self):
        self.assertIn(GRAPH_BACKEND, {"langgraph", "fallback"})


if __name__ == "__main__":
    unittest.main()
