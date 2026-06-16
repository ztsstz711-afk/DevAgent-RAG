import unittest
import tempfile
from unittest.mock import patch

from src.graph_builder import build_graph
from src.rag_pipeline import RAGPipeline
from src.reranker import LexicalReranker
from src.retriever import TfidfRetriever


class StaticRetriever:
    backend = "static"

    def __init__(self, documents):
        self.documents = documents

    def search(self, query, top_k=4, min_score=0.01, code_only=False):
        return [dict(document) for document in self.documents[:top_k]]


class LexicalRerankerTests(unittest.TestCase):
    def setUp(self):
        self.reranker = LexicalReranker()

    def test_empty_documents_returns_empty_list(self):
        self.assertEqual(self.reranker.rerank("OpenAI rate limits", []), [])

    def test_top_k_limits_results(self):
        documents = [
            {"content": "OpenAI rate limits", "chunk_id": "a"},
            {"content": "CUDA memory", "chunk_id": "b"},
            {"content": "LangChain retriever", "chunk_id": "c"},
        ]

        result = self.reranker.rerank("OpenAI", documents, top_k=2)

        self.assertEqual(len(result), 2)

    def test_keyword_match_ranks_document_first(self):
        documents = [
            {"content": "PyTorch DataLoader batches tensors", "title": "Data loading", "chunk_id": "pytorch"},
            {"content": "OpenAI rate limits use exponential backoff", "title": "Rate limits", "chunk_id": "openai"},
        ]

        result = self.reranker.rerank("How to handle OpenAI rate limits?", documents)

        self.assertEqual(result[0]["chunk_id"], "openai")

    def test_error_type_match_ranks_error_document_first(self):
        documents = [
            {"content": "General Python package installation guide", "chunk_id": "general", "score": 0.9},
            {
                "content": "ModuleNotFoundError and ImportError usually mean a missing dependency.",
                "chunk_id": "module_error",
                "score": 0.1,
            },
        ]

        result = self.reranker.rerank("ModuleNotFoundError: No module named transformers", documents)

        self.assertEqual(result[0]["chunk_id"], "module_error")

    def test_original_document_fields_are_preserved(self):
        document = {
            "content": "CUDA out of memory reduce batch size",
            "source": "cuda.md",
            "chunk_id": "chunk_001",
            "score": 0.7,
            "metadata": {"kept": True},
        }

        result = self.reranker.rerank("CUDA OOM", [document])

        self.assertEqual(result[0], document)
        self.assertIn("metadata", result[0])
        self.assertNotIn("rerank_score", result[0])

    def test_graph_default_disabled_keeps_retrieval_order(self):
        documents = [
            {
                "content": "General Python package installation guide",
                "source": "general.md",
                "product": "python",
                "chunk_id": "general",
                "score": 0.9,
            },
            {
                "content": "ModuleNotFoundError for transformers means the dependency is missing.",
                "source": "module.md",
                "product": "huggingface",
                "chunk_id": "module_error",
                "score": 0.1,
            },
        ]
        graph = build_graph(
            StaticRetriever(documents),
            {"retrieval": {"top_k": 2, "min_score": 0.0}, "quality": {"min_citations": 1}},
        )

        result = graph.invoke({"question": "ModuleNotFoundError: No module named transformers", "tool_trace": []})

        self.assertEqual([item["chunk_id"] for item in result["documents"]], ["general", "module_error"])
        self.assertNotIn("reranker_info", result)

    def test_graph_enabled_reranks_retrieved_documents(self):
        documents = [
            {
                "content": "General Python package installation guide",
                "source": "general.md",
                "product": "python",
                "chunk_id": "general",
                "score": 0.9,
            },
            {
                "content": "ModuleNotFoundError and ImportError for transformers mean a missing dependency.",
                "source": "module.md",
                "product": "huggingface",
                "chunk_id": "module_error",
                "score": 0.1,
            },
        ]
        graph = build_graph(
            StaticRetriever(documents),
            {
                "retrieval": {"top_k": 2, "min_score": 0.0},
                "reranker": {"enabled": True, "mode": "lexical", "top_k": 2},
                "quality": {"min_citations": 1},
            },
        )

        result = graph.invoke({"question": "ModuleNotFoundError: No module named transformers", "tool_trace": []})

        self.assertEqual(result["documents"][0]["chunk_id"], "module_error")
        self.assertEqual(
            result["reranker_info"],
            {
                "reranker_enabled": True,
                "reranker_mode": "lexical",
                "original_retrieved_count": 2,
                "reranked_count": 2,
            },
        )
        self.assertTrue(result["tool_trace"][2]["output"]["reranker_enabled"])

    def test_pipeline_reranker_preserves_no_evidence_refusal(self):
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
            "reranker": {"enabled": True, "mode": "lexical", "top_k": 2},
            "quality": {"min_citations": 1},
        }

        with patch("src.rag_pipeline.load_config", return_value=config):
            result = RAGPipeline(retriever=retriever).ask("How to deploy Kubernetes on AWS?", write_trace=False)

        self.assertEqual(result["answer"], "未在当前文档知识库中找到明确依据。")
        self.assertFalse(result["quality"]["passed"])
        self.assertIn("no_evidence", result["quality"]["issues"])
        self.assertTrue(result["reranker_info"]["reranker_enabled"])

    def test_tfidf_retrieval_with_reranker_works(self):
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
        with tempfile.TemporaryDirectory() as temp:
            config = {
                "paths": {"index": "unused", "output": temp},
                "retrieval": {"top_k": 2, "min_score": 0.0},
                "reranker": {"enabled": True, "mode": "lexical", "top_k": 2},
                "quality": {"min_citations": 1},
            }

            with patch("src.rag_pipeline.load_config", return_value=config):
                result = RAGPipeline(retriever=retriever).ask("How to handle OpenAI rate limits?")

        self.assertTrue(result["quality"]["passed"])
        self.assertEqual(result["documents"][0]["product"], "openai")
        self.assertEqual(result["reranker_info"]["reranked_count"], 2)


if __name__ == "__main__":
    unittest.main()
