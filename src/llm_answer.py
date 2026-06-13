from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

from .answer_generator import NO_EVIDENCE_ANSWER, citation, generate_answer


@dataclass
class AnswerResult:
    answer: str
    backend: str
    fallback_reason: str | None = None


class LLMAnswerGenerator:
    """Generate grounded answers with an optional OpenAI-compatible backend."""

    def __init__(self, mode: str = "template", config: dict | None = None, timeout: float = 30.0):
        config = config or {}
        self.requested_mode = mode
        self.api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.base_url_env = config.get("base_url_env", "OPENAI_BASE_URL")
        self.model_env = config.get("model_env", "OPENAI_MODEL")
        self.api_key = os.getenv(self.api_key_env, "")
        self.base_url = os.getenv(self.base_url_env, "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv(self.model_env, "gpt-4o-mini")
        self.timeout = timeout

    def build_prompt(self, question: str, evidence: list[dict], error_info: dict | None = None) -> str:
        evidence_blocks = []
        for item in evidence:
            evidence_blocks.append(f"{citation(item)}\n{item['content']}")
        error_context = json.dumps(error_info, ensure_ascii=False) if error_info else "None"
        return f"""You are DevAgent-RAG, a grounded AI development assistant.

Rules:
1. Answer only from the EVIDENCE below. Do not invent or add information outside it.
2. Every key conclusion must include the exact citation supplied with its evidence.
3. If evidence is insufficient, answer exactly: {NO_EVIDENCE_ANSWER}
4. Use clear Markdown with these sections: ## Answer, ## Evidence, ## Next Steps.

QUESTION:
{question}

ERROR CONTEXT:
{error_context}

EVIDENCE:
{chr(10).join(evidence_blocks) if evidence_blocks else '(none)'}
"""

    def generate(
        self,
        question: str,
        evidence: list[dict],
        error_info: dict | None = None,
        code_snippets: list[dict] | None = None,
    ) -> AnswerResult:
        template = lambda: generate_answer(question, evidence, error_info, code_snippets)
        if not evidence:
            return AnswerResult(template(), "template", "no_evidence")
        if self.requested_mode != "openai":
            return AnswerResult(template(), "template")
        if not self.api_key:
            reason = f"{self.api_key_env} is not set; falling back to template answer mode."
            print(f"LLM answer fallback: {reason}", file=sys.stderr)
            return AnswerResult(template(), "template", reason)
        try:
            answer = self._call_openai(self.build_prompt(question, evidence, error_info))
            return AnswerResult(answer, "openai")
        except Exception as exc:  # Network and compatible API errors must not break the RAG pipeline.
            reason = f"OpenAI-compatible API request failed: {type(exc).__name__}: {exc}"
            print(f"LLM answer fallback: {reason}", file=sys.stderr)
            return AnswerResult(template(), "template", reason)

    def _call_openai(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Answer strictly from supplied evidence and preserve citations."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        answer = data["choices"][0]["message"]["content"].strip()
        if not answer:
            raise ValueError("API returned an empty answer")
        return answer
