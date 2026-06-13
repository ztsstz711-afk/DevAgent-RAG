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
    "CUDA out of memory while training a model",
    "ModuleNotFoundError: No module named 'transformers'",
    "How should I configure LLaMAFactory for a small GPU?",
]


def run_demo() -> Path:
    prepare_sample_docs()
    build_index()
    pipeline = RAGPipeline()
    sections = [
        "# DevAgent-RAG Demo Results\n",
        f"Graph backend: `{pipeline.graph_backend}`  \nRetrieval mode: `{pipeline.retrieval_mode}`  \nRetriever backend: `{pipeline.retriever_backend}`\n",
    ]
    for index, question in enumerate(QUESTIONS, start=1):
        result = pipeline.ask(question)
        sections.append(f"## Case {index}: {question}\n\nRoute: `{result['route']}`  \nQuality: `{result['quality']['score']:.2f}`\n\n{result['answer']}\n")
    target = project_path("data/output/demo_results.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(sections), encoding="utf-8")
    return target


if __name__ == "__main__":
    output = run_demo()
    pipeline = RAGPipeline()
    print(f"Graph backend: {pipeline.graph_backend}")
    print(f"Retrieval mode: {pipeline.retrieval_mode}")
    print(f"Retriever backend: {pipeline.retriever_backend}")
    print(f"Demo complete: {output}")
