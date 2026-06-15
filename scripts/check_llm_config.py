from __future__ import annotations

import os


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def mask_api_key(api_key: str) -> str:
    if not api_key:
        return "not configured"
    prefix = api_key[:3]
    suffix = api_key[-4:] if len(api_key) >= 4 else api_key[-1:]
    return f"{prefix}***{suffix}"


def get_llm_config(environ: dict[str, str] | None = None) -> dict:
    env = os.environ if environ is None else environ
    api_key = env.get("OPENAI_API_KEY", "").strip()
    return {
        "api_key_configured": bool(api_key),
        "masked_api_key": mask_api_key(api_key),
        "base_url": env.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL,
        "model": env.get("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
    }


def format_llm_config(config: dict) -> str:
    lines = [
        "DevAgent-RAG LLM configuration",
        f"OPENAI_API_KEY configured: {'yes' if config['api_key_configured'] else 'no'}",
        f"OPENAI_API_KEY preview: {config['masked_api_key']}",
        f"OPENAI_BASE_URL: {config['base_url']}",
        f"OPENAI_MODEL: {config['model']}",
    ]
    if config["api_key_configured"]:
        lines.append("Configuration is ready. Run: python scripts/run_llm_demo.py")
    else:
        lines.append("No API key found. LLM Answer will fall back to template answer mode.")
    lines.append("This check does not send any network request.")
    return "\n".join(lines)


def check_llm_config(environ: dict[str, str] | None = None) -> dict:
    config = get_llm_config(environ)
    print(format_llm_config(config))
    return config


if __name__ == "__main__":
    check_llm_config()
