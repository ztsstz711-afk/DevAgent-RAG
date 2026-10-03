"""Evaluate TF-IDF, embedding, and hybrid retrieval on frozen v8 qrels.

A requested embedding or hybrid run that falls back to TF-IDF is reported as
unavailable, not as an embedding or hybrid score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_index import build_index
from src.rag_pipeline import RAGPipeline
from src.utils import project_path, write_json


MODES = ("tfidf", "embedding", "hybrid")
DEFAULT_CONFIG = "configs/official_retrieval_v8.yaml"
DEFAULT_QUERIES = ROOT / "data" / "evals" / "official_issue_queries_v8.json"
DEFAULT_QRELS = ROOT / "data" / "evals" / "official_issue_qrels_v8.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(queries_path: Path = DEFAULT_QUERIES, qrels_path: Path = DEFAULT_QRELS) -> tuple[dict, dict]:
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    items = queries.get("queries", [])
    ids = [item.get("id") for item in items]
    if not items or len(set(ids)) != len(ids) or None in ids:
        raise ValueError("Queries must be non-empty and have unique IDs")
    if set(ids) != set(qrels.get("qrels", {})):
        raise ValueError("Query IDs and qrel IDs must match exactly")
    for item in items:
        relevant = qrels["qrels"][item["id"]]
        if not isinstance(relevant, list):
            raise ValueError(f"Qrel for {item['id']} must be a list")
        if item["should_refuse"] != (not relevant):
            raise ValueError(f"Refusal label and qrel cardinality disagree for {item['id']}")
    return queries, qrels


def _source_id(item: dict) -> str:
    return Path(item.get("source", "")).stem


def _metrics(details: list[dict]) -> dict:
    positive = [item for item in details if item["relevant_source_ids"]]
    if not positive:
        return {"positive_queries": 0}
    ranks = [item["first_relevant_rank"] for item in positive]
    def recall_at(k: int) -> float:
        return round(sum(bool(set(item["retrieved_source_ids"][:k]) & set(item["relevant_source_ids"])) for item in positive) / len(positive), 3)
    return {
        "positive_queries": len(positive),
        "hit_at_1": recall_at(1),
        "hit_at_3": recall_at(3),
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        "recall_at_5": recall_at(5),
        "mrr": round(sum(1 / rank if rank else 0 for rank in ranks) / len(positive), 3),
    }


def _evaluate_mode(mode: str, config_path: str, items: list[dict], qrels: dict) -> dict:
    pipeline = RAGPipeline(config_path=config_path, retrieval_mode=mode, top_k=5)
    if pipeline.retrieval_mode != mode:
        return {
            "requested_mode": mode,
            "status": "unavailable_fallback",
            "effective_mode": pipeline.retrieval_mode,
            "backend": pipeline.retriever_backend,
            "fallback_reason": pipeline.retrieval_fallback_reason,
            "details": [],
        }
    details = []
    refusals = 0
    for query in items:
        results = pipeline.retriever.search(query["query"], top_k=5, min_score=0.0)
        source_ids = list(dict.fromkeys(_source_id(result) for result in results))
        relevant = qrels[query["id"]]
        first_rank = next((rank for rank, source_id in enumerate(source_ids, start=1) if source_id in relevant), None)
        evidence = pipeline.ask(query["query"], write_trace=False).get("evidence_assessment", {})
        refused = not evidence.get("valid", False)
        if query["should_refuse"]:
            refusals += refused
        details.append({
            "id": query["id"],
            "origin_type": query["origin_type"],
            "origin_url": query["origin_url"],
            "relevant_source_ids": relevant,
            "retrieved_source_ids": source_ids,
            "first_relevant_rank": first_rank,
            "should_refuse": query["should_refuse"],
            "refused": refused,
            "evidence_issues": evidence.get("issues", []),
        })
    negatives = [item for item in details if item["should_refuse"]]
    return {
        "requested_mode": mode,
        "status": "completed",
        "effective_mode": pipeline.retrieval_mode,
        "backend": pipeline.retriever_backend,
        "fallback_reason": None,
        "metrics": _metrics(details),
        "refusal_queries": len(negatives),
        "refusal_rate": round(refusals / len(negatives), 3) if negatives else None,
        "details": details,
    }


def _write_markdown(report: dict, target: Path) -> None:
    lines = [
        f"# Official Corpus Retrieval Experiment: {report['protocol']['id']}",
        "",
        "This is a source-level retrieval experiment over a frozen, versioned official-document corpus. It does not measure answer correctness, user preference, or production quality.",
        "",
        f"- Documents: {report['corpus']['documents']}; chunks: {report['corpus']['chunks']}",
        f"- Queries: {report['protocol']['queries']}; source-retrieval queries: {report['protocol']['positive_queries']}; refusal controls: {report['protocol']['refusal_queries']}",
        "- A requested mode with `unavailable_fallback` has no score; its TF-IDF fallback is not reported as embedding or hybrid.",
        "",
        "| Requested mode | Status | Effective mode | Backend | Hit@1 | Hit@3 | Recall@5 | MRR | Refusal rate |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        result = report["modes"][mode]
        metrics = result.get("metrics", {})
        number = lambda key: f"{metrics[key]:.1%}" if key in metrics else "n/a"
        mrr = f"{metrics['mrr']:.3f}" if "mrr" in metrics else "n/a"
        refusal = f"{result['refusal_rate']:.1%}" if result.get("refusal_rate") is not None else "n/a"
        lines.append(
            f"| {mode} | {result['status']} | {result['effective_mode']} | {result['backend']} | "
            f"{number('hit_at_1')} | {number('hit_at_3')} | {number('recall_at_5')} | {mrr} | {refusal} |"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(config_path: str = DEFAULT_CONFIG, queries_path: Path = DEFAULT_QUERIES, qrels_path: Path = DEFAULT_QRELS) -> dict:
    queries, qrels = load_protocol(queries_path, qrels_path)
    chunks, index_path = build_index(config_path=config_path)
    config = RAGPipeline(config_path=config_path).config
    corpus_dir = project_path(config["external_docs"]["imported_docs_dir"])
    manifest_path = corpus_dir / config["external_docs"].get("manifest_filename", "corpus_manifest_v8.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing frozen corpus manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = queries["queries"]
    report = {
        "protocol": {
            "id": queries["dataset_id"],
            "queries": len(items),
            "positive_queries": sum(not item["should_refuse"] for item in items),
            "refusal_queries": sum(item["should_refuse"] for item in items),
            "query_file_sha256": _sha256(queries_path),
            "qrels_file_sha256": _sha256(qrels_path),
        },
        "corpus": {
            "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": _sha256(manifest_path),
            "documents": len(manifest["entries"]),
            "chunks": chunks,
            "index": str(index_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "modes": {mode: _evaluate_mode(mode, config_path, items, qrels["qrels"]) for mode in MODES},
    }
    output_dir = project_path(config["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "official_retrieval_experiment_v8.json", report)
    _write_markdown(report, output_dir / "official_retrieval_experiment_v8.md")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    args = parser.parse_args()
    report = evaluate(args.config, args.queries, args.qrels)
    for mode, result in report["modes"].items():
        print(f"{mode}: {result['status']} ({result['effective_mode']}, {result['backend']})")
