from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_index import build_index
from scripts.evaluate import CASES
from scripts.prepare_sample_docs import prepare_sample_docs
from src.rag_pipeline import RAGPipeline
from src.utils import project_path, write_json


MODES = ("tfidf", "embedding", "hybrid")


def _evaluate_mode(mode: str) -> dict:
    pipeline = RAGPipeline(retrieval_mode=mode)
    supported = [case for case in CASES if case["product"] is not None]
    hit_1 = 0
    hit_3 = 0
    non_empty = 0
    details = []
    for case in CASES:
        results = pipeline.retriever.search(
            case["question"],
            top_k=3,
            min_score=pipeline.config["retrieval"].get("min_score", 0.05),
        )
        products = [item.get("product") for item in results]
        non_empty += bool(results)
        if case["product"] is not None:
            hit_1 += bool(products) and products[0] == case["product"]
            hit_3 += case["product"] in products[:3]
        details.append({
            "question": case["question"],
            "expected_product": case["product"],
            "top_products": products,
            "non_empty": bool(results),
        })
    return {
        "requested_mode": mode,
        "effective_mode": pipeline.retrieval_mode,
        "retriever_backend": pipeline.retriever_backend,
        "fallback_reason": pipeline.retrieval_fallback_reason,
        "hit_at_1": round(hit_1 / len(supported), 3),
        "hit_at_3": round(hit_3 / len(supported), 3),
        "retrieval_non_empty_rate": round(non_empty / len(CASES), 3),
        "details": details,
    }


def _write_markdown(report: dict) -> Path:
    lines = [
        "# Retrieval Mode Evaluation",
        "",
        "This comparison uses the built-in sample documents and does not represent production quality.",
        "",
        "| Requested mode | Effective mode | Backend | Hit@1 | Hit@3 | Non-empty rate | Fallback |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for mode in MODES:
        item = report["modes"][mode]
        reason = (item["fallback_reason"] or "none").replace("|", "\\|")
        lines.append(
            f"| {mode} | {item['effective_mode']} | {item['retriever_backend']} | "
            f"{item['hit_at_1']:.1%} | {item['hit_at_3']:.1%} | "
            f"{item['retrieval_non_empty_rate']:.1%} | {reason} |"
        )
    target = project_path("data/output/retrieval_eval.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def retrieval_eval() -> dict:
    prepare_sample_docs()
    build_index()
    report = {"total_cases": len(CASES), "modes": {mode: _evaluate_mode(mode) for mode in MODES}}
    write_json("data/output/retrieval_eval.json", report)
    _write_markdown(report)
    return report


if __name__ == "__main__":
    report = retrieval_eval()
    for mode in MODES:
        item = report["modes"][mode]
        print(
            f"{mode}: effective={item['effective_mode']}, backend={item['retriever_backend']}, "
            f"Hit@1={item['hit_at_1']:.1%}, Hit@3={item['hit_at_3']:.1%}, "
            f"non_empty={item['retrieval_non_empty_rate']:.1%}"
        )
        if item["fallback_reason"]:
            print(f"  fallback: {item['fallback_reason']}")
    print(f"JSON report: {project_path('data/output/retrieval_eval.json')}")
    print(f"Markdown report: {project_path('data/output/retrieval_eval.md')}")
