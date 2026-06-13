from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from scripts.build_index import build_index
from scripts.evaluate import evaluate
from scripts.prepare_external_docs import prepare_external_docs
from scripts.retrieval_eval import retrieval_eval
from src.document_loader import SUPPORTED_EXTENSIONS
from src.utils import load_config, project_path


class WebOperationError(RuntimeError):
    pass


def display_task_type(result: dict, question: str = "") -> str:
    """Map backend routing fields to stable Web UI task type labels."""
    current = result.get("task_type") or result.get("route") or "question"
    if current in {"doc_qa", "error_debug", "code_lookup", "config_help"}:
        return current

    text = question.lower()
    error_type = result.get("error_info", {}).get("primary_type", "")
    config_terms = ("api key", "api_key", "openai_api_key", "environment variable", "env var", "configuration", "configure")
    code_terms = ("code example", "example code", "code snippet", "sample code", "show me code")
    if error_type == "openai_api_key_missing" or any(term in text for term in config_terms):
        return "config_help"
    if any(term in text for term in code_terms):
        return "code_lookup"
    if current in {"error", "debug"} or result.get("error_info", {}).get("is_error"):
        return "error_debug"
    return "doc_qa"


def save_uploaded_file(uploaded_file, destination_dir: str | Path | None = None) -> Path:
    config = load_config().get("web", {})
    root = project_path(destination_dir or config.get("uploaded_docs_dir", "data/docs_uploaded"))
    filename = Path(getattr(uploaded_file, "name", "")).name
    if not filename or Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only .md, .mdx, .txt, and .ipynb uploads are supported")
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    if b"\x00" in data and Path(filename).suffix.lower() != ".ipynb":
        raise ValueError("Binary files are not supported")
    root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    target.write_bytes(data)
    return target


def validate_github_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        return False
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        return False
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or any(part in {".", ".."} for part in parts):
        return False
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts):
        return False
    repo_name = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    return bool(repo_name and repo_name not in {".", ".."})


def clone_github_repo(url: str, external_dir: str | Path | None = None) -> dict:
    if not validate_github_url(url):
        raise ValueError("Repository URL must be an HTTPS GitHub URL like https://github.com/org/repo")
    config = load_config()
    if not config.get("web", {}).get("allow_github_clone", True):
        raise WebOperationError("GitHub cloning is disabled by configuration")
    root = project_path(external_dir or config.get("external_docs", {}).get("external_dir", "external"))
    repo_name = Path(urlparse(url.strip()).path).name
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    destination = root / repo_name
    if destination.exists():
        raise WebOperationError(f"Destination already exists: {destination}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            ["git", "clone", "--depth", "1", url.strip(), str(destination)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WebOperationError(f"Could not start git clone: {exc}") from exc
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "Unknown git error").strip()
        raise WebOperationError(f"Git clone failed: {message}")
    return {"url": url.strip(), "repo_name": repo_name, "path": str(destination)}


def load_json_report(path: str | Path) -> dict:
    target = project_path(path)
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WebOperationError(f"Could not read JSON report {target}: {exc}") from exc


def run_prepare_external_docs() -> dict:
    return prepare_external_docs()


def run_build_index() -> dict:
    chunk_count, index_path = build_index()
    stats = load_json_report("data/output/index_stats.json")
    return {"chunk_count": chunk_count, "index_path": str(index_path), "stats": stats}


def run_evaluate() -> dict:
    return evaluate()


def run_retrieval_eval() -> dict:
    return retrieval_eval()
