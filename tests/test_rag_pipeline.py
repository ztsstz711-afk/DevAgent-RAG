import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.rag_pipeline import RAGPipeline
from src.retriever import TfidfRetriever


class RagPipelineTests(unittest.TestCase):
    def setUp(self):
        chunks = [
            {"content": "OpenAI rate limits should use exponential backoff.\n```python\ntime.sleep(2)\n```", "source": "rate_limits.md", "product": "openai", "chunk_id": "chunk_001"},
            {"content": "CUDA OOM can be fixed by reducing batch size.", "source": "cuda.md", "product": "pytorch", "chunk_id": "chunk_001"},
        ]
        self.retriever = TfidfRetriever().build(chunks)

    def test_end_to_end_answer_has_citation_and_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            config = {
                "paths": {"index": "unused", "output": temp},
                "retrieval": {"top_k": 2, "min_score": 0.0},
                "quality": {"min_citations": 1},
            }
            with patch("src.rag_pipeline.load_config", return_value=config):
                pipeline = RAGPipeline(retriever=self.retriever)
                result = pipeline.ask("How to handle OpenAI rate limits?")
            self.assertIn("[openai | rate_limits.md | chunk_001]", result["answer"])
            self.assertTrue(result["quality"]["passed"])
            self.assertTrue((Path(temp) / "tool_trace.jsonl").exists())
            self.assertIn(pipeline.graph_backend, {"langgraph", "fallback"})
            self.assertIn(pipeline.retriever_backend, {"sklearn_tfidf", "fallback"})

    def test_unsupported_query_returns_no_evidence_refusal(self):
        config = {
            "paths": {"index": "unused", "output": "data/output"},
            "retrieval": {"top_k": 2, "min_score": 0.05},
            "quality": {"min_citations": 1},
        }
        with patch("src.rag_pipeline.load_config", return_value=config):
            result = RAGPipeline(retriever=self.retriever).ask(
                "How to deploy Kubernetes on AWS?", write_trace=False
            )
        self.assertEqual(result["answer"], "未在当前文档知识库中找到明确依据。")
        self.assertFalse(result["quality"]["passed"])
        self.assertIn("no_evidence", result["quality"]["issues"])

    def test_insurance_query_returns_out_of_domain_refusal(self):
        config = {
            "paths": {"index": "unused", "output": "data/output"},
            "retrieval": {"top_k": 2, "min_score": 0.0},
            "quality": {"min_citations": 1},
        }
        with patch("src.rag_pipeline.load_config", return_value=config):
            result = RAGPipeline(retriever=self.retriever).ask(
                "How to price an insurance product?", write_trace=False
            )
        self.assertEqual(result["answer"], "未在当前文档知识库中找到明确依据。")
        self.assertIn("out_of_domain", result["quality"]["issues"])

    def test_supported_llamafactory_and_cuda_queries_still_answer(self):
        chunks = [
            {"content": "LLaMAFactory LoRA SFT dataset uses instruction input output fields.", "source": "dataset.md", "product": "llamafactory", "chunk_id": "chunk_001"},
            {"content": "CUDA out of memory can be handled by reducing batch size.", "source": "cuda.md", "product": "pytorch", "chunk_id": "chunk_001"},
        ]
        retriever = TfidfRetriever().build(chunks)
        config = {
            "paths": {"index": "unused", "output": "data/output"},
            "retrieval": {"top_k": 2, "min_score": 0.0},
            "quality": {"min_citations": 1},
        }
        with patch("src.rag_pipeline.load_config", return_value=config):
            pipeline = RAGPipeline(retriever=retriever)
            lora = pipeline.ask("LLaMAFactory LoRA SFT dataset format", write_trace=False)
            cuda = pipeline.ask("CUDA out of memory", write_trace=False)
        self.assertTrue(lora["quality"]["passed"])
        self.assertTrue(cuda["quality"]["passed"])


if __name__ == "__main__":
    unittest.main()
