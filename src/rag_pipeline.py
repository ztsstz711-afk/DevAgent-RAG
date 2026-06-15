from __future__ import annotations

from pathlib import Path
import sys

from .embedding_retriever import EmbeddingRetriever, EmbeddingUnavailableError
from .answer_generator import NO_EVIDENCE_ANSWER
from .graph_builder import GRAPH_BACKEND, build_graph
from .hybrid_retriever import HybridRetriever
from .retriever import RETRIEVER_BACKEND, TfidfRetriever
from .utils import append_jsonl, load_config, project_path


class RAGPipeline:
    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        retriever=None,
        llm_mode: str | None = None,
        retrieval_mode: str | None = None,
        top_k: int | None = None,
    ):
        self.config = load_config(config_path)
        if llm_mode is not None:
            self.config.setdefault("llm_answer", {})["enabled"] = llm_mode == "openai"
            self.config["llm_answer"]["mode"] = llm_mode
        retrieval_config = self.config.setdefault("retrieval", {})
        if top_k is not None:
            retrieval_config["top_k"] = int(top_k)
        requested_mode = retrieval_mode or retrieval_config.get("mode", "tfidf")
        retrieval_config["mode"] = requested_mode
        index_path = project_path(self.config["paths"]["index"])
        tfidf_retriever = retriever or TfidfRetriever.load(index_path)
        self.retrieval_requested_mode = requested_mode
        self.retrieval_fallback_reason = None
        if retriever is not None:
            self.retriever = retriever
            self.retrieval_mode = requested_mode
            self.retriever_backend = getattr(retriever, "backend", RETRIEVER_BACKEND)
        else:
            self.retriever = self._select_retriever(tfidf_retriever, requested_mode, retrieval_config)
        self.graph = build_graph(self.retriever, self.config)
        self.graph_backend = GRAPH_BACKEND

    def _select_retriever(self, tfidf_retriever, mode: str, config: dict):
        if mode == "tfidf":
            self.retrieval_mode = "tfidf"
            self.retriever_backend = RETRIEVER_BACKEND
            return tfidf_retriever
        if mode not in {"embedding", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        try:
            embedding = EmbeddingRetriever(config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
            embedding.build_index(tfidf_retriever.chunks)
            if mode == "embedding":
                self.retrieval_mode = "embedding"
                self.retriever_backend = embedding.backend
                return embedding
            hybrid = HybridRetriever(tfidf_retriever, embedding, alpha=float(config.get("hybrid_alpha", 0.5)))
            self.retrieval_mode = "hybrid"
            self.retriever_backend = hybrid.backend
            return hybrid
        except EmbeddingUnavailableError as exc:
            self.retrieval_fallback_reason = str(exc)
            self.retrieval_mode = "tfidf"
            self.retriever_backend = f"{RETRIEVER_BACKEND}_fallback"
            print(f"Retrieval fallback: {exc}", file=sys.stderr)
            return tfidf_retriever

    def ask(self, question: str, write_trace: bool = True) -> dict:
        result = self.graph.invoke({"question": question, "tool_trace": []})
        assessment = result.get("evidence_assessment", {})
        if assessment and not assessment.get("valid", False):
            result["answer"] = NO_EVIDENCE_ANSWER
            result["quality"]["passed"] = False
            result["quality"]["citations"] = []
            result["quality"]["issues"] = list(dict.fromkeys([
                *result["quality"].get("issues", []), *assessment.get("issues", []), "no_evidence"
            ]))
        if write_trace:
            trace_path = Path(self.config["paths"]["output"]) / "tool_trace.jsonl"
            append_jsonl(trace_path, {"question": question, "route": result["route"], "trace": result["tool_trace"]})
        return result
