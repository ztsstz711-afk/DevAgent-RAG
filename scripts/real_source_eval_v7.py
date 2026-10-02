"""Run a small, frozen smoke evaluation against traceable v7 sources.

This is deliberately separate from the project's built-in sample
evaluation.  It reports source-specific retrieval and refusal behavior rather
than presenting the result as a broad production benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag_pipeline import RAGPipeline
from src.utils import project_path, write_json

CASES_PATH = ROOT / "data" / "evals" / "real_source_cases_v7.json"


def load_cases(cases_path: Path = CASES_PATH) -> tuple[dict, list[dict]]:
    dataset = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = dataset.get("cases", [])
    if not cases:
        raise ValueError(f"No cases found in {cases_path}")
    required = {"id", "question", "expected_source", "source_url", "source_type", "should_refuse"}
    for case in cases:
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"Case {case.get('id', '<unknown>')} is missing fields: {', '.join(missing)}")
    return dataset, cases


def _refused(result: dict) -> bool:
    quality = result.get("quality", {})
    return "no_evidence" in quality.get("issues", []) and not quality.get("passed", False)


def _retrieval_metrics(details: list[dict]) -> dict:
    positive = [item for item in details if item["expected_source"]]
    ranks = [item.get("expected_source_rank") for item in positive]
    return {
        "expected_source_cases": len(positive),
        "hit_at_1": sum(rank == 1 for rank in ranks),
        "hit_at_3": sum(rank is not None and rank <= 3 for rank in ranks),
        "mrr": round(
            sum(1 / rank for rank in ranks if rank is not None) / len(positive), 3
        ) if positive else None,
    }


def _write_markdown(report: dict, output_stem: str) -> Path:
    retrieval = report["retrieval_metrics"]
    lines = [
        "# DevAgent-RAG v7 Real-Source Smoke Evaluation",
        "",
        "This is an eight-case, source-traceable smoke evaluation. It is not a production benchmark.",
        "",
        f"- Graph backend: `{report['graph_backend']}`",
        f"- Retrieval mode: `{report['retrieval_mode']}`",
        f"- Retriever backend: `{report['retriever_backend']}`",
        f"- Overall pass rate: {report['pass_rate']:.1%} ({report['passed_cases']}/{report['total_cases']})",
        f"- Expected-source Hit@1: {retrieval['hit_at_1']}/{retrieval['expected_source_cases']}",
        f"- Expected-source Hit@3: {retrieval['hit_at_3']}/{retrieval['expected_source_cases']}",
        f"- Expected-source MRR: {retrieval['mrr']:.3f}" if retrieval["mrr"] is not None else "- Expected-source MRR: n/a (no source-retrieval cases)",
        "",
        "| ID | Expected behavior | Result | Expected-source rank | Evidence status | Retrieved sources |",
        "|---|---|---:|---:|---|---|",
    ]
    for item in report["details"]:
        expected = "refuse without direct source" if item["should_refuse"] else f"retrieve `{item['expected_source']}`"
        sources = ", ".join(item["retrieved_sources"]) or "(none)"
        lines.append(
            f"| {item['id']} | {expected} | {'PASS' if item['passed'] else 'FAIL'} | "
            f"{item['expected_source_rank'] or '-'} | "
            f"{item['evidence_valid']} ({', '.join(item['evidence_issues']) or 'none'}) | {sources} |"
        )
    target = project_path(f"data/output/{output_stem}.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def evaluate(
    config_path: str = "configs/real_source_eval_v7.yaml",
    output_stem: str = "real_source_eval_v7",
    cases_path: Path = CASES_PATH,
) -> dict:
    dataset, cases = load_cases(cases_path)
    pipeline = RAGPipeline(config_path=config_path)
    details = []
    for case in cases:
        result = pipeline.ask(case["question"], write_trace=False)
        assessment = result.get("evidence_assessment", {})
        retrieved_sources = [item.get("source", "") for item in result.get("documents", [])]
        expected_source = case["expected_source"]
        expected_source_rank = (
            retrieved_sources.index(expected_source) + 1
            if expected_source in retrieved_sources
            else None
        )
        refused = _refused(result)
        if case["should_refuse"]:
            passed = refused
        else:
            passed = bool(assessment.get("valid")) and case["expected_source"] in retrieved_sources
        details.append({
            "id": case["id"],
            "question": case["question"],
            "source_url": case["source_url"],
            "expected_source": case["expected_source"],
            "expected_source_rank": expected_source_rank,
            "should_refuse": case["should_refuse"],
            "passed": passed,
            "refused": refused,
            "evidence_valid": bool(assessment.get("valid")),
            "evidence_issues": assessment.get("issues", []),
            "keyword_overlap": assessment.get("keyword_overlap"),
            "retrieved_sources": retrieved_sources,
            "retrieved_citations": [
                f"{item.get('product', '')}/{item.get('source', '')}/{item.get('chunk_id', '')}"
                for item in result.get("documents", [])
            ],
            "answer_backend": result.get("answer_backend"),
        })
    report = {
        "evaluation_type": "frozen_real_source_smoke_test",
        "dataset_id": dataset["dataset_id"],
        "dataset_path": str(cases_path.resolve().relative_to(ROOT)).replace("\\", "/"),
        "config_path": config_path.replace("\\", "/"),
        "scope": dataset.get("scope", "Dataset-defined source and evaluation scope."),
        "graph_backend": pipeline.graph_backend,
        "retrieval_mode": pipeline.retrieval_mode,
        "retriever_backend": pipeline.retriever_backend,
        "total_cases": len(details),
        "passed_cases": sum(item["passed"] for item in details),
        "pass_rate": round(sum(item["passed"] for item in details) / len(details), 3),
        "details": details,
    }
    report["retrieval_metrics"] = _retrieval_metrics(details)
    write_json(f"data/output/{output_stem}.json", report)
    _write_markdown(report, output_stem)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/real_source_eval_v7.yaml")
    parser.add_argument("--output-stem", default="real_source_eval_v7")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    args = parser.parse_args()
    report = evaluate(config_path=args.config, output_stem=args.output_stem, cases_path=args.cases)
    retrieval = report["retrieval_metrics"]
    print(f"Real-source smoke pass rate: {report['pass_rate']:.1%} ({report['passed_cases']}/{report['total_cases']})")
    print(
        "Expected-source retrieval: "
        f"Hit@1 {retrieval['hit_at_1']}/{retrieval['expected_source_cases']}, "
        f"Hit@3 {retrieval['hit_at_3']}/{retrieval['expected_source_cases']}, "
        f"MRR {retrieval['mrr']:.3f}" if retrieval["mrr"] is not None else "MRR n/a"
    )
    for item in report["details"]:
        print(f"{item['id']}: {'PASS' if item['passed'] else 'FAIL'}")
