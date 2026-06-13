from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from src.rag_pipeline import RAGPipeline
from src.utils import load_config, project_path
from src.web_utils import (
    WebOperationError,
    clone_github_repo,
    display_task_type,
    load_json_report,
    run_build_index,
    run_evaluate,
    run_prepare_external_docs,
    run_retrieval_eval,
    save_uploaded_file,
    validate_github_url,
)


st.set_page_config(page_title="DevAgent-RAG", page_icon="D", layout="wide")
CONFIG = load_config()


@st.cache_resource(show_spinner=False)
def get_pipeline(retrieval_mode: str, top_k: int, answer_mode: str, index_version: float):
    del index_version
    return RAGPipeline(retrieval_mode=retrieval_mode, top_k=top_k, llm_mode=answer_mode)


def index_version() -> float:
    path = project_path(CONFIG["paths"]["index"])
    return path.stat().st_mtime if path.exists() else 0.0


def show_index_stats() -> None:
    stats = load_json_report("data/output/index_stats.json")
    if not stats:
        st.info("No index statistics yet. Build the index first.")
        return
    columns = st.columns(4)
    columns[0].metric("Documents", stats.get("total_documents", 0))
    columns[1].metric("Chunks", stats.get("total_chunks", 0))
    columns[2].metric("Imported", stats.get("imported_docs_count", 0))
    columns[3].metric("Uploaded", stats.get("uploaded_docs_count", 0))
    st.write("Sources", stats.get("sources", []))
    st.json(stats.get("chunks_per_source", {}))


def knowledge_base_page() -> None:
    st.header("Knowledge Base")
    uploads = st.file_uploader(
        "Upload documentation",
        type=["md", "mdx", "txt", "ipynb"],
        accept_multiple_files=True,
    )
    if st.button("Save Uploaded Files", disabled=not uploads):
        try:
            saved = [str(save_uploaded_file(item)) for item in uploads]
            st.success(f"Saved {len(saved)} file(s). Build the index to use them.")
            st.code("\n".join(saved))
        except (ValueError, OSError) as exc:
            st.error(str(exc))

    st.subheader("GitHub Documentation Repository")
    repo_url = st.text_input("Repository URL", placeholder="https://github.com/org/repo")
    if repo_url and not validate_github_url(repo_url):
        st.warning("Only URLs like https://github.com/org/repo are allowed.")
    if st.button("Clone Repo", disabled=not repo_url or not validate_github_url(repo_url)):
        try:
            with st.spinner("Cloning repository..."):
                result = clone_github_repo(repo_url)
            st.success(f"Cloned {result['repo_name']} to {result['path']}")
        except (ValueError, WebOperationError) as exc:
            st.error(str(exc))

    action_columns = st.columns(2)
    if action_columns[0].button("Import External Docs", use_container_width=True):
        try:
            with st.spinner("Importing supported documents..."):
                result = run_prepare_external_docs()
            st.success(f"Imported {result['total_imported']} document(s).")
            st.json(result)
        except Exception as exc:
            st.error(f"External document import failed: {exc}")

    if action_columns[1].button("Build Index", use_container_width=True):
        try:
            with st.spinner("Building TF-IDF index and statistics..."):
                result = run_build_index()
            get_pipeline.clear()
            st.success(f"Built index with {result['chunk_count']} chunks.")
        except Exception as exc:
            st.error(f"Index build failed: {exc}")
    st.subheader("Index Statistics")
    show_index_stats()


def ask_page(retrieval_mode: str, top_k: int, answer_mode: str) -> None:
    st.header("Ask / Debug")
    question = st.text_area(
        "Technical question or error log",
        height=180,
        placeholder="CUDA out of memory when training a PyTorch model",
    )
    if st.button("Run Agent", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Running LangGraph agent..."):
                pipeline = get_pipeline(retrieval_mode, top_k, answer_mode, index_version())
                result = pipeline.ask(question.strip())
            st.subheader("Result")
            summary = st.columns(4)
            summary[0].metric("task_type", display_task_type(result, question))
            summary[1].metric("retrieval_mode", pipeline.retrieval_mode)
            summary[2].metric("retriever_backend", pipeline.retriever_backend)
            summary[3].metric("answer_backend", result.get("answer_backend", "template"))
            if pipeline.retrieval_fallback_reason:
                st.warning(pipeline.retrieval_fallback_reason)
            st.markdown(result.get("answer", ""))

            quality_tab, evidence_tab, chunks_tab, trace_tab = st.tabs(
                ["Quality Report", "Evidence Citations", "Retrieved Chunks", "Tool Trace"]
            )
            with quality_tab:
                st.json(result.get("quality", {}))
            with evidence_tab:
                citations = result.get("quality", {}).get("citations", [])
                st.write(citations if citations else "No evidence citations.")
            with chunks_tab:
                for index, chunk in enumerate(result.get("documents", []), start=1):
                    label = f"{index}. {chunk.get('product')} / {chunk.get('source')} / {chunk.get('chunk_id')} ({chunk.get('score', 0):.3f})"
                    with st.expander(label):
                        st.write(chunk.get("text", chunk.get("content", "")))
                        st.json({key: chunk.get(key) for key in ("title", "section_title", "source_path", "has_code")})
            with trace_tab:
                st.json(result.get("tool_trace", []))
        except FileNotFoundError:
            st.error("Index not found. Build the index from the Knowledge Base page first.")
        except Exception as exc:
            st.error(f"Agent run failed: {exc}")


def _report_preview(json_path: str, markdown_path: str) -> None:
    report = load_json_report(json_path)
    if report:
        st.json(report)
    markdown_file = project_path(markdown_path)
    if markdown_file.exists():
        content = markdown_file.read_text(encoding="utf-8")
        st.download_button("Download Markdown", content, file_name=markdown_file.name, mime="text/markdown")
        with st.expander("Markdown preview"):
            st.markdown(content)


def evaluation_page() -> None:
    st.header("Evaluation")
    actions = st.columns(2)
    if actions[0].button("Run Evaluation", use_container_width=True):
        with st.spinner("Running agent evaluation..."):
            run_evaluate()
        get_pipeline.clear()
        st.success("Evaluation complete.")
    if actions[1].button("Run Retrieval Eval", use_container_width=True):
        with st.spinner("Comparing retrieval modes..."):
            run_retrieval_eval()
        get_pipeline.clear()
        st.success("Retrieval evaluation complete.")

    eval_report = load_json_report("data/output/eval_report.json")
    if eval_report:
        st.subheader("Agent Evaluation Metrics")
        metric_names = (
            "route_accuracy", "retrieval_hit_rate", "citation_rate",
            "quality_pass_rate", "no_evidence_count", "tool_success_rate",
        )
        columns = st.columns(3)
        for index, name in enumerate(metric_names):
            value = eval_report.get(name, 0)
            display = f"{value:.1%}" if isinstance(value, float) and name != "no_evidence_count" else value
            columns[index % 3].metric(name, display)

    retrieval_report = load_json_report("data/output/retrieval_eval.json")
    if retrieval_report:
        st.subheader("Retrieval Comparison")
        rows = []
        for mode, item in retrieval_report.get("modes", {}).items():
            rows.append({
                "mode": mode,
                "effective_mode": item.get("effective_mode"),
                "backend": item.get("retriever_backend"),
                "Hit@1": item.get("hit_at_1"),
                "Hit@3": item.get("hit_at_3"),
                "non_empty": item.get("retrieval_non_empty_rate"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    eval_tab, retrieval_tab = st.tabs(["Eval Report", "Retrieval Eval Report"])
    with eval_tab:
        _report_preview("data/output/eval_report.json", "data/output/eval_report.md")
    with retrieval_tab:
        _report_preview("data/output/retrieval_eval.json", "data/output/retrieval_eval.md")


st.title("DevAgent-RAG Local Console")
st.caption("Local Streamlit control panel backed by the existing src/ and scripts/ modules.")

with st.sidebar:
    st.header("Configuration")
    retrieval_mode = st.selectbox("Retrieval mode", ["tfidf", "embedding", "hybrid"])
    top_k = st.slider("Top K", min_value=1, max_value=10, value=int(CONFIG["retrieval"].get("top_k", 3)))
    answer_mode = st.selectbox("Answer mode", ["template", "openai"])
    page = st.radio("Page", ["Knowledge Base", "Ask / Debug", "Evaluation"])
    st.divider()
    try:
        pipeline = get_pipeline(retrieval_mode, top_k, answer_mode, index_version())
        st.write(f"Graph backend: `{pipeline.graph_backend}`")
        st.write(f"Retriever backend: `{pipeline.retriever_backend}`")
        st.write(f"Retrieval mode: `{pipeline.retrieval_mode}`")
        if pipeline.retrieval_fallback_reason:
            st.warning(pipeline.retrieval_fallback_reason)
    except Exception as exc:
        st.warning(f"Backend unavailable until an index is built: {exc}")
    st.write(f"OPENAI_API_KEY configured: `{'yes' if os.getenv('OPENAI_API_KEY') else 'no'}`")

if page == "Knowledge Base":
    knowledge_base_page()
elif page == "Ask / Debug":
    ask_page(retrieval_mode, top_k, answer_mode)
else:
    evaluation_page()
