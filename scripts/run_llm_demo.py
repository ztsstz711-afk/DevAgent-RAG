from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_index import build_index
from scripts.prepare_sample_docs import prepare_sample_docs
from src.rag_pipeline import RAGPipeline
from src.utils import project_path


QUESTIONS = [
    "How to handle OpenAI rate limits?",
    "CUDA out of memory when training a PyTorch model",
    "How to deploy Kubernetes on AWS?",
]


def run_llm_demo() -> Path:
    prepare_sample_docs()
    build_index()
    pipeline = RAGPipeline(llm_mode="openai")
    sections = [
        "# DevAgent-RAG LLM Answer Demo",
        "",
        f"Graph backend: `{pipeline.graph_backend}`  ",
        f"Retrieval mode: `{pipeline.retrieval_mode}`  ",
        f"Retriever backend: `{pipeline.retriever_backend}`",
        "",
    ]
    for index, question in enumerate(QUESTIONS, start=1):
        result = pipeline.ask(question)
        sections.extend([
            f"## Case {index}: {question}",
            "",
            f"Answer backend: `{result['answer_backend']}`  ",
            f"Fallback reason: `{result.get('answer_fallback_reason') or 'none'}`  ",
            f"Quality passed: `{result['quality']['passed']}`",
            "",
            result["answer"],
            "",
        ])
    target = project_path("data/output/llm_demo_results.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(sections), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(f"LLM demo complete: {run_llm_demo()}")
