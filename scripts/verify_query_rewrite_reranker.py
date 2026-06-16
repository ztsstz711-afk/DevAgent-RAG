from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_pipeline import RAGPipeline
from src.retriever import TfidfRetriever


DOCUMENTS = [
    {
        "content": "General Python package installation guide for virtual environments and pip.",
        "source": "python_install.md",
        "product": "python",
        "chunk_id": "chunk_001",
    },
    {
        "content": "ModuleNotFoundError and ImportError for transformers mean a missing dependency.",
        "source": "module_errors.md",
        "product": "huggingface",
        "chunk_id": "chunk_002",
    },
    {
        "content": "AutoTokenizer.from_pretrained loads HuggingFace tokenizer files.",
        "source": "tokenizer.md",
        "product": "huggingface",
        "chunk_id": "chunk_003",
    },
    {
        "content": "CUDA out of memory can be handled by reducing batch size or gradient accumulation.",
        "source": "cuda_oom.md",
        "product": "pytorch",
        "chunk_id": "chunk_004",
    },
]

ERROR_QUERY = """Traceback (most recent call last):
  File "demo.py", line 1, in <module>
    from transformers import AutoTokenizer
ModuleNotFoundError: No module named 'transformers'
"""

UNSUPPORTED_QUERY = "How to deploy Kubernetes on AWS?"

SETTINGS = [
    ("A. rewrite=false, reranker=false", False, False),
    ("B. rewrite=true, reranker=false", True, False),
    ("C. rewrite=true, reranker=true", True, True),
]


def make_config(output_dir: str, query_rewrite_enabled: bool, reranker_enabled: bool) -> dict:
    return {
        "paths": {"index": "unused", "output": output_dir},
        "retrieval": {"mode": "tfidf", "top_k": 3, "min_score": 0.0},
        "query_rewrite": {"enabled": query_rewrite_enabled, "mode": "rule_based"},
        "reranker": {"enabled": reranker_enabled, "mode": "lexical", "top_k": 3},
        "quality": {"min_citations": 1},
        "llm_answer": {"enabled": False, "mode": "template"},
    }


def run_case(question: str, query_rewrite_enabled: bool, reranker_enabled: bool) -> dict:
    retriever = TfidfRetriever().build(copy.deepcopy(DOCUMENTS))
    with tempfile.TemporaryDirectory() as temp:
        config = make_config(temp, query_rewrite_enabled, reranker_enabled)
        with patch("src.rag_pipeline.load_config", return_value=config):
            return RAGPipeline(retriever=retriever).ask(question, write_trace=False)


def answer_preview(answer: str) -> list[str]:
    return answer.splitlines()[:5]


def retrieved_order(result: dict) -> list[str]:
    return [
        f"{doc.get('product')} | {doc.get('source')} | {doc.get('chunk_id')} | score={doc.get('score')}"
        for doc in result.get("documents", [])
    ]


def print_result(label: str, result: dict, unsupported_result: dict) -> None:
    rewrite_info = result.get("query_rewrite_info")
    reranker_info = result.get("reranker_info")
    print(f"\n## {label}")
    print("answer preview:")
    for line in answer_preview(result["answer"]):
        print(f"  {line}")
    print(f"query_rewrite_info exists: {bool(rewrite_info)}")
    print(f"search_query: {(rewrite_info or {}).get('search_query')}")
    print(f"reranker_info exists: {bool(reranker_info)}")
    print(f"reranker_info: {reranker_info}")
    print("retrieved document order:")
    for index, item in enumerate(retrieved_order(result), start=1):
        print(f"  {index}. {item}")
    print("unsupported query check:")
    print(f"  answer: {unsupported_result['answer']}")
    print(f"  quality_passed: {unsupported_result['quality']['passed']}")
    print(f"  issues: {unsupported_result['quality']['issues']}")


def main() -> None:
    print("# Query Rewrite + Reranker Verification")
    print("Error query:")
    print(ERROR_QUERY.strip())
    print(f"Unsupported query: {UNSUPPORTED_QUERY}")

    for label, rewrite_enabled, reranker_enabled in SETTINGS:
        result = run_case(ERROR_QUERY, rewrite_enabled, reranker_enabled)
        unsupported_result = run_case(UNSUPPORTED_QUERY, rewrite_enabled, reranker_enabled)
        print_result(label, result, unsupported_result)


if __name__ == "__main__":
    main()
