import unittest

from src.retriever import TfidfRetriever
from src.tools import code_snippet_finder_tool, doc_search_tool, error_parser_tool, quality_check_tool


class ToolTests(unittest.TestCase):
    def setUp(self):
        chunks = [
            {"content": "Rate limit backoff", "source": "rate.md", "product": "openai", "chunk_id": "chunk_001"},
            {"content": "```python\ntime.sleep(2)\n```", "source": "code.md", "product": "openai", "chunk_id": "chunk_002"},
        ]
        self.retriever = TfidfRetriever().build(chunks)

    def test_four_tools(self):
        self.assertEqual(doc_search_tool("rate limit", self.retriever, top_k=1)[0]["source"], "rate.md")
        self.assertEqual(len(code_snippet_finder_tool("sleep", self.retriever)), 1)
        self.assertTrue(error_parser_tool("CUDA out of memory")["is_error"])
        answer = "A sufficiently detailed answer about retry behavior. [openai | rate.md | chunk_001]"
        self.assertTrue(quality_check_tool(answer)["passed"])


if __name__ == "__main__":
    unittest.main()
