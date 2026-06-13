from __future__ import annotations

try:
    from langgraph.graph import END, START, StateGraph
    GRAPH_BACKEND = "langgraph"
except ImportError:  # The real LangGraph path is used whenever dependencies are installed.
    from .compat import END, START, SimpleStateGraph as StateGraph
    GRAPH_BACKEND = "fallback"

from .agent_state import AgentState
from .llm_answer import LLMAnswerGenerator
from .tools import code_snippet_finder_tool, doc_search_tool, error_parser_tool, quality_check_tool


def _trace(state: AgentState, tool: str, output: object) -> list[dict]:
    return [*state.get("tool_trace", []), {"tool": tool, "output": output}]


def build_graph(retriever, config: dict | None = None):
    config = config or {}
    retrieval = config.get("retrieval", {})
    quality = config.get("quality", {})
    llm_config = config.get("llm_answer", {})
    answer_mode = llm_config.get("mode", "template") if llm_config.get("enabled", False) else "template"
    answer_generator = LLMAnswerGenerator(mode=answer_mode, config=llm_config)

    def route_task(state: AgentState) -> dict:
        parsed = error_parser_tool(state["question"])
        route = "error" if parsed["is_error"] else "question"
        return {"route": route, "tool_trace": _trace(state, "route_task", {"route": route})}

    def parse_error_node(state: AgentState) -> dict:
        result = error_parser_tool(state["question"])
        return {"error_info": result, "tool_trace": _trace(state, "error_parser_tool", result)}

    def retrieve_docs(state: AgentState) -> dict:
        result = doc_search_tool(
            state["question"], retriever,
            top_k=retrieval.get("top_k", 4), min_score=retrieval.get("min_score", 0.01),
        )
        compact = [{"citation": f"{x['product']}/{x['source']}/{x['chunk_id']}", "score": x["score"]} for x in result]
        return {"documents": result, "tool_trace": _trace(state, "doc_search_tool", compact)}

    def find_code_snippets(state: AgentState) -> dict:
        preferred_sources = [item["source"] for item in state.get("documents", [])]
        result = code_snippet_finder_tool(state["question"], retriever, top_k=2, preferred_sources=preferred_sources)
        compact = [{"citation": f"{x['product']}/{x['source']}/{x['chunk_id']}", "score": x["score"]} for x in result]
        return {"code_snippets": result, "tool_trace": _trace(state, "code_snippet_finder_tool", compact)}

    def generate_answer_node(state: AgentState) -> dict:
        generated = answer_generator.generate(
            state["question"], state.get("documents", []), state.get("error_info"), state.get("code_snippets", [])
        )
        metadata = {
            "characters": len(generated.answer),
            "backend": generated.backend,
            "fallback_reason": generated.fallback_reason,
        }
        return {
            "answer": generated.answer,
            "answer_backend": generated.backend,
            "answer_fallback_reason": generated.fallback_reason,
            "tool_trace": _trace(state, "generate_answer", metadata),
        }

    def check_quality_node(state: AgentState) -> dict:
        result = quality_check_tool(
            state["answer"],
            min_citations=quality.get("min_citations", 1),
            no_evidence=not state.get("documents"),
        )
        return {"quality": result, "tool_trace": _trace(state, "quality_check_tool", result)}

    graph = StateGraph(AgentState)
    graph.add_node("route_task", route_task)
    graph.add_node("parse_error", parse_error_node)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("find_code_snippets", find_code_snippets)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("check_quality", check_quality_node)
    graph.add_edge(START, "route_task")
    graph.add_conditional_edges("route_task", lambda state: state["route"], {"error": "parse_error", "question": "retrieve_docs"})
    graph.add_edge("parse_error", "retrieve_docs")
    graph.add_edge("retrieve_docs", "find_code_snippets")
    graph.add_edge("find_code_snippets", "generate_answer")
    graph.add_edge("generate_answer", "check_quality")
    graph.add_edge("check_quality", END)
    return graph.compile()
