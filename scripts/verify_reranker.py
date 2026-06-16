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
        "content": "ModuleNotFoundError and ImportError usually mean a missing dependency such as transformers.",
        "source": "module_errors.md",
        "product": "huggingface",
        "chunk_id": "chunk_002",
    },
    {
        "content": "CUDA out of memory can be handled by reducing batch size or gradient accumulation.",
        "source": "cuda_oom.md",
        "product": "pytorch",
        "chunk_id": "chunk_003",
    },
]


def make_config(output_dir: str, reranker_enabled: bool) -> dict:
    return {
        "paths": {"index": "unused", "output": output_dir},
        "retrieval": {"top_k": 3, "min_score": 0.0},
        "reranker": {"enabled": reranker_enabled, "mode": "lexical", "top_k": 3},
        "quality": {"min_citations": 1},
        "llm_answer": {"enabled": False, "mode": "template"},
    }


def run_case(question: str, reranker_enabled: bool) -> dict:
    retriever = TfidfRetriever().build(copy.deepcopy(DOCUMENTS))
    with tempfile.TemporaryDirectory() as temp:
        config = make_config(temp, reranker_enabled)
        with patch("src.rag_pipeline.load_config", return_value=config):
            return RAGPipeline(retriever=retriever).ask(question, write_trace=False)


def summarize(label: str, result: dict) -> None:
    print(f"\n## {label}")
    print("answer preview:")
    for line in result["answer"].splitlines()[:5]:
        print(f"  {line}")
    print("retrieved order:")
    for index, doc in enumerate(result.get("documents", []), start=1):
        print(
            f"  {index}. {doc.get('product')} | {doc.get('source')} | "
            f"{doc.get('chunk_id')} | score={doc.get('score')}"
        )
    trace = next((item for item in result.get("tool_trace", []) if item.get("tool") == "doc_search_tool"), {})
    output = trace.get("output", {})
    print(f"trace has reranker_info: {'reranker_enabled' in output}")
    print(f"state reranker_info: {result.get('reranker_info')}")


def main() -> None:
    question = "Python package installation guide pip ModuleNotFoundError transformers"
    disabled = run_case(question, reranker_enabled=False)
    enabled = run_case(question, reranker_enabled=True)

    print(f"# Reranker Verification\nQuestion: {question}")
    summarize("reranker disabled", disabled)
    summarize("reranker enabled", enabled)

    unsupported_question = "How to deploy Kubernetes on AWS?"
    unsupported = run_case(unsupported_question, reranker_enabled=True)
    print(f"\n## unsupported query with reranker enabled")
    print(f"question: {unsupported_question}")
    print(f"answer: {unsupported['answer']}")
    print(f"quality passed: {unsupported['quality']['passed']}")
    print(f"quality issues: {unsupported['quality']['issues']}")
    print(f"reranker_info: {unsupported.get('reranker_info')}")


if __name__ == "__main__":
    main()
