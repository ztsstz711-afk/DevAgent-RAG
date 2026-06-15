from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_index import build_index
from scripts.prepare_sample_docs import prepare_sample_docs
from src.rag_pipeline import RAGPipeline
from src.utils import project_path, write_json


CASES = [
    {"question": "How do I create OpenAI embeddings?", "product": "openai", "route": "question"},
    {"question": "How to handle OpenAI rate limits?", "product": "openai", "route": "question"},
    {"question": "How do OpenAI structured outputs enforce a schema?", "product": "openai", "route": "question"},
    {"question": "How does a LangChain retriever work?", "product": "langchain", "route": "question"},
    {"question": "How do I use LangChain tool calling?", "product": "langchain", "route": "question"},
    {"question": "How should I configure a PyTorch DataLoader?", "product": "pytorch", "route": "question"},
    {"question": "PyTorch CUDA out of memory", "product": "pytorch", "route": "error"},
    {"question": "How do I load a HuggingFace tokenizer?", "product": "huggingface", "route": "question"},
    {"question": "How do I start the vLLM OpenAI server?", "product": "vllm", "route": "question"},
    {"question": "What is the LLaMAFactory LoRA SFT dataset format?", "product": "llamafactory", "route": "question"},
    {"question": "ModuleNotFoundError: No module named transformers", "product": "huggingface", "route": "error"},
    {"question": "OPENAI_API_KEY missing", "product": "openai", "route": "error"},
    {"question": "RuntimeError: dtype mismatch", "product": "pytorch", "route": "error"},
    {"question": "How to deploy Kubernetes on AWS?", "product": None, "route": "question"},
    {"question": "How to price an insurance product?", "product": None, "route": "question"},
]


def _tool_success(result: dict) -> bool:
    called = {entry["tool"] for entry in result.get("tool_trace", [])}
    required = {"route_task", "doc_search_tool", "code_snippet_finder_tool", "generate_answer", "quality_check_tool"}
    if result.get("route") == "error":
        required.add("error_parser_tool")
    return required.issubset(called)


def _write_markdown(report: dict) -> Path:
    lines = [
        "# DevAgent-RAG Evaluation Report",
        "",
        f"- Graph backend: `{report['graph_backend']}`",
        f"- Retrieval mode: `{report['retrieval_mode']}`",
        f"- Retriever backend: `{report['retriever_backend']}`",
        f"- Total cases: {report['total_cases']}",
        f"- Route accuracy: {report['route_accuracy']:.1%}",
        f"- Retrieval hit rate: {report['retrieval_hit_rate']:.1%}",
        f"- Citation rate: {report['citation_rate']:.1%}",
        f"- Quality pass rate: {report['quality_pass_rate']:.1%}",
        f"- No-evidence count: {report['no_evidence_count']}",
        f"- Unsupported query count: {report['unsupported_query_count']}",
        f"- Tool success rate: {report['tool_success_rate']:.1%}",
        f"- Hit@1: {report['hit_at_1']:.1%}",
        f"- Hit@3: {report['hit_at_3']:.1%}",
        "",
        "## Cases",
        "",
        "| Question | Expected product | Route | Retrieval | Quality | No evidence |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in report["details"]:
        lines.append(
            f"| {item['question']} | {item['expected_product'] or 'unsupported'} | {item['actual_route']} | "
            f"{item['retrieval_hit']} | {item['quality_passed']} | {item['no_evidence']} |"
        )
    target = project_path("data/output/eval_report.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def evaluate() -> dict:
    prepare_sample_docs()
    build_index()
    pipeline = RAGPipeline()
    details = []
    for case in CASES:
        result = pipeline.ask(case["question"], write_trace=False)
        ranked_products = [item["product"] for item in result.get("documents", [])]
        supported = case["product"] is not None
        retrieval_non_empty = bool(result.get("documents"))
        valid_evidence = result.get("evidence_assessment", {}).get("valid", False)
        retrieval_hit = supported and valid_evidence and case["product"] in ranked_products
        no_evidence = "no_evidence" in result["quality"]["issues"]
        unsupported_refusal = not supported and no_evidence and not result["quality"]["passed"]
        details.append({
            "question": case["question"],
            "expected_product": case["product"],
            "expected_route": case["route"],
            "actual_route": result["route"],
            "route_hit": result["route"] == case["route"],
            "retrieval_non_empty": retrieval_non_empty,
            "valid_evidence_hit": retrieval_hit,
            "retrieval_hit": retrieval_hit,
            "citation_hit": bool(result["quality"]["citations"]),
            "quality_passed": result["quality"]["passed"],
            "no_evidence": no_evidence,
            "unsupported_refusal": unsupported_refusal,
            "tool_success": _tool_success(result),
            "hit_at_1": supported and bool(ranked_products) and ranked_products[0] == case["product"],
            "hit_at_3": supported and case["product"] in ranked_products[:3],
        })

    total = len(details)
    supported_details = [item for item in details if item["expected_product"] is not None]
    supported_count = len(supported_details)
    report = {
        "graph_backend": pipeline.graph_backend,
        "retrieval_mode": pipeline.retrieval_mode,
        "retriever_backend": pipeline.retriever_backend,
        "total_cases": total,
        "route_accuracy": round(sum(item["route_hit"] for item in details) / total, 3),
        "retrieval_hit_rate": round(sum(item["retrieval_hit"] for item in supported_details) / supported_count, 3),
        "citation_rate": round(sum(item["citation_hit"] for item in supported_details) / supported_count, 3),
        "quality_pass_rate": round(sum(item["quality_passed"] for item in details) / total, 3),
        "no_evidence_count": sum(item["no_evidence"] for item in details),
        "unsupported_query_count": total - supported_count,
        "unsupported_refusal_count": sum(item["unsupported_refusal"] for item in details),
        "unsupported_refusal_rate": round(
            sum(item["unsupported_refusal"] for item in details) / max(1, total - supported_count), 3
        ),
        "tool_success_rate": round(sum(item["tool_success"] for item in details) / total, 3),
        "hit_at_1": round(sum(item["hit_at_1"] for item in supported_details) / supported_count, 3),
        "hit_at_3": round(sum(item["hit_at_3"] for item in supported_details) / supported_count, 3),
        "details": details,
    }
    write_json("data/output/eval_report.json", report)
    _write_markdown(report)
    return report


if __name__ == "__main__":
    report = evaluate()
    print(f"Graph backend: {report['graph_backend']}")
    print(f"Retrieval mode: {report['retrieval_mode']}")
    print(f"Retriever backend: {report['retriever_backend']}")
    print(f"Total cases: {report['total_cases']}")
    for name in ("route_accuracy", "retrieval_hit_rate", "citation_rate", "quality_pass_rate", "tool_success_rate", "hit_at_1", "hit_at_3"):
        print(f"{name}: {report[name]:.1%}")
    print(f"no_evidence_count: {report['no_evidence_count']}")
    print(f"unsupported_query_count: {report['unsupported_query_count']}")
    print(f"JSON report: {project_path('data/output/eval_report.json')}")
    print(f"Markdown report: {project_path('data/output/eval_report.md')}")
