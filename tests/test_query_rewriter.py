import unittest
from unittest.mock import patch

from src.graph_builder import build_graph
from src.query_rewriter import QueryRewriter
from src.rag_pipeline import RAGPipeline
from src.retriever import TfidfRetriever


class StaticRetriever:
    backend = "static"

    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def search(self, query, top_k=4, min_score=0.01, code_only=False):
        self.queries.append(query)
        return [dict(document) for document in self.documents[:top_k]]


class QueryRewriterTests(unittest.TestCase):
    def setUp(self):
        self.rewriter = QueryRewriter()

    def test_regular_question_is_not_over_rewritten(self):
        query = "How do LangChain retrievers work with OpenAI embeddings?"

        result = self.rewriter.rewrite(query)

        self.assertEqual(result["original_query"], query)
        self.assertEqual(result["search_query"], query)
        self.assertFalse(result["rewrite_applied"])
        self.assertEqual(result["rewrite_reason"], "none")

    def test_extracts_library_names(self):
        result = self.rewriter.rewrite("Use PyTorch, HuggingFace transformers, vLLM, and LLaMA-Factory")

        self.assertIn("pytorch", result["library_names"])
        self.assertIn("huggingface", result["library_names"])
        self.assertIn("transformers", result["library_names"])
        self.assertIn("vllm", result["library_names"])
        self.assertIn("llamafactory", result["library_names"])

    def test_extracts_api_and_class_names(self):
        result = self.rewriter.rewrite("How to use ChatOpenAI, AutoTokenizer.from_pretrained, and StateGraph?")

        self.assertIn("ChatOpenAI", result["api_names"])
        self.assertIn("AutoTokenizer", result["api_names"])
        self.assertIn("from_pretrained", result["api_names"])
        self.assertIn("StateGraph", result["api_names"])

    def test_module_not_found_extracts_error_type_and_module_name(self):
        result = self.rewriter.rewrite("ModuleNotFoundError: No module named 'transformers'", route="error")

        self.assertEqual(result["error_type"], "ModuleNotFoundError")
        self.assertEqual(result["module_name"], "transformers")
        self.assertTrue(result["rewrite_applied"])
        self.assertEqual(result["search_query"], "ModuleNotFoundError transformers installation import error")

    def test_cuda_oom_generates_short_search_query(self):
        traceback = """
        Traceback (most recent call last):
          File "train.py", line 20, in <module>
        RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB on GPU 0.
        """

        result = self.rewriter.rewrite(traceback, route="error")

        self.assertEqual(result["error_type"], "CUDA out of memory")
        self.assertEqual(result["search_query"], "CUDA out of memory pytorch memory batch size")
        self.assertLess(len(result["search_query"]), len(traceback))

    def test_openai_api_key_missing_extracts_error_type(self):
        result = self.rewriter.rewrite("OPENAI_API_KEY is missing or not set", route="error")

        self.assertEqual(result["error_type"], "OPENAI_API_KEY missing")
        self.assertEqual(
            result["search_query"],
            "OPENAI_API_KEY missing openai api key authentication environment variable",
        )

    def test_empty_or_none_input_is_safe(self):
        for query in ("", None):
            with self.subTest(query=query):
                result = self.rewriter.rewrite(query)
                self.assertEqual(result["original_query"], query or "")
                self.assertEqual(result["search_query"], "")
                self.assertEqual(result["keywords"], [])
                self.assertIsNone(result["error_type"])
                self.assertIsNone(result["module_name"])
                self.assertFalse(result["rewrite_applied"])

    def test_graph_default_disabled_keeps_original_query(self):
        documents = [
            {
                "content": "ModuleNotFoundError transformers installation import error",
                "source": "module.md",
                "product": "huggingface",
                "chunk_id": "chunk_001",
                "score": 0.9,
            }
        ]
        retriever = StaticRetriever(documents)
        graph = build_graph(retriever, {"retrieval": {"top_k": 1, "min_score": 0.0}})
        query = "ModuleNotFoundError: No module named 'transformers'"

        result = graph.invoke({"question": query, "tool_trace": []})

        self.assertEqual(retriever.queries[0], query)
        self.assertNotIn("query_rewrite_info", result)

    def test_graph_enabled_records_rewrite_info_and_uses_search_query(self):
        documents = [
            {
                "content": "ModuleNotFoundError transformers installation import error",
                "source": "module.md",
                "product": "huggingface",
                "chunk_id": "chunk_001",
                "score": 0.9,
            }
        ]
        retriever = StaticRetriever(documents)
        graph = build_graph(
            retriever,
            {
                "retrieval": {"top_k": 1, "min_score": 0.0},
                "query_rewrite": {"enabled": True, "mode": "rule_based"},
            },
        )

        result = graph.invoke({"question": "ModuleNotFoundError: No module named 'transformers'", "tool_trace": []})

        self.assertEqual(retriever.queries[0], "ModuleNotFoundError transformers installation import error")
        self.assertTrue(result["query_rewrite_info"]["query_rewrite_enabled"])
        self.assertEqual(result["query_rewrite_info"]["module_name"], "transformers")
        self.assertEqual(result["query_rewrite_info"]["rewrite_reason"], "module_import_error")
        self.assertTrue(result["tool_trace"][2]["output"]["query_rewrite_enabled"])

    def test_pipeline_long_module_error_rewrite_compresses_search_query(self):
        traceback = """
        Traceback (most recent call last):
          File "demo.py", line 1, in <module>
            import transformers
        ModuleNotFoundError: No module named 'transformers'
        """
        documents = [
            {
                "content": "ModuleNotFoundError transformers installation import error",
                "source": "module.md",
                "product": "huggingface",
                "chunk_id": "chunk_001",
                "score": 0.9,
            }
        ]
        retriever = StaticRetriever(documents)
        config = {
            "paths": {"index": "unused", "output": "data/output"},
            "retrieval": {"top_k": 1, "min_score": 0.0},
            "query_rewrite": {"enabled": True, "mode": "rule_based"},
            "quality": {"min_citations": 1},
        }

        with patch("src.rag_pipeline.load_config", return_value=config):
            result = RAGPipeline(retriever=retriever).ask(traceback, write_trace=False)

        self.assertEqual(retriever.queries[0], "ModuleNotFoundError transformers installation import error")
        self.assertLess(len(result["query_rewrite_info"]["search_query"]), len(traceback))

    def test_pipeline_rewrite_preserves_unsupported_refusal(self):
        documents = [
            {
                "content": "OpenAI rate limits should use exponential backoff.",
                "source": "rate.md",
                "product": "openai",
                "chunk_id": "chunk_001",
            },
            {
                "content": "CUDA out of memory can be handled by reducing batch size.",
                "source": "cuda.md",
                "product": "pytorch",
                "chunk_id": "chunk_002",
            },
        ]
        retriever = TfidfRetriever().build(documents)
        config = {
            "paths": {"index": "unused", "output": "data/output"},
            "retrieval": {"top_k": 2, "min_score": 0.0},
            "query_rewrite": {"enabled": True, "mode": "rule_based"},
            "quality": {"min_citations": 1},
        }

        with patch("src.rag_pipeline.load_config", return_value=config):
            result = RAGPipeline(retriever=retriever).ask("How to deploy Kubernetes on AWS?", write_trace=False)

        self.assertEqual(result["answer"], "未在当前文档知识库中找到明确依据。")
        self.assertFalse(result["quality"]["passed"])
        self.assertIn("out_of_domain", result["quality"]["issues"])
        self.assertTrue(result["query_rewrite_info"]["query_rewrite_enabled"])

    def test_query_rewrite_and_reranker_work_together(self):
        documents = [
            {
                "content": "General Python package installation guide for virtual environments and pip.",
                "source": "python_install.md",
                "product": "python",
                "chunk_id": "chunk_001",
                "score": 0.9,
            },
            {
                "content": "ModuleNotFoundError and ImportError for transformers mean a missing dependency.",
                "source": "module.md",
                "product": "huggingface",
                "chunk_id": "chunk_002",
                "score": 0.1,
            },
        ]
        graph = build_graph(
            StaticRetriever(documents),
            {
                "retrieval": {"top_k": 2, "min_score": 0.0},
                "query_rewrite": {"enabled": True, "mode": "rule_based"},
                "reranker": {"enabled": True, "mode": "lexical", "top_k": 2},
                "quality": {"min_citations": 1},
            },
        )

        result = graph.invoke({"question": "ModuleNotFoundError: No module named transformers", "tool_trace": []})

        self.assertEqual(result["documents"][0]["chunk_id"], "chunk_002")
        self.assertTrue(result["query_rewrite_info"]["query_rewrite_enabled"])
        self.assertTrue(result["reranker_info"]["reranker_enabled"])


if __name__ == "__main__":
    unittest.main()
