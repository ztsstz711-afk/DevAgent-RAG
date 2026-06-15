from __future__ import annotations

import re


SUPPORTED_TERMS = {
    "ai", "llm", "model", "openai", "langchain", "langgraph", "pytorch", "torch",
    "huggingface", "transformers", "tokenizer", "vllm", "llamafactory", "rag",
    "retrieval", "retriever", "embedding", "embeddings", "agent", "tool", "tools",
    "sft", "lora", "finetune", "finetuning", "training", "inference", "serving",
    "dataloader", "cuda", "gpu", "batch", "dtype", "module", "python", "api",
    "key", "rate", "limits", "structured", "outputs", "dataset", "quantization",
}
OUT_OF_DOMAIN_GROUPS = {
    "cloud_infrastructure": {"kubernetes", "k8s", "aws", "azure", "gcp", "eks", "ec2", "terraform", "cloudformation"},
    "insurance": {"insurance", "premium", "actuarial", "underwriting", "policyholder"},
    "finance": {"stock", "stocks", "bond", "bonds", "portfolio", "dividend", "mortgage"},
    "geography": {"geography", "capital", "continent", "latitude", "longitude"},
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "the",
    "this", "to", "use", "using", "what", "when", "where", "which", "with", "you",
}
MIN_KEYWORD_OVERLAP = 0.1


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def assess_evidence(question: str, documents: list[dict], min_score: float = 0.05) -> dict:
    query_tokens = _tokens(question)
    matched_out_of_domain = sorted(
        name for name, terms in OUT_OF_DOMAIN_GROUPS.items() if query_tokens & terms
    )
    supported_terms = sorted(query_tokens & SUPPORTED_TERMS)
    out_of_domain = bool(matched_out_of_domain and not supported_terms)

    top_score = max((float(item.get("score", 0.0)) for item in documents), default=0.0)
    evidence_text = " ".join(
        f"{item.get('product', '')} {item.get('source', '')} {item.get('title', '')} "
        f"{item.get('section_title', '')} {item.get('text', item.get('content', ''))}"
        for item in documents[:3]
    )
    evidence_tokens = _tokens(evidence_text)
    meaningful_query = query_tokens - {"example", "help", "handle", "problem", "issue"}
    overlap_tokens = sorted(meaningful_query & evidence_tokens)
    overlap_ratio = len(overlap_tokens) / max(1, len(meaningful_query))

    issues = []
    if not documents:
        issues.append("no_evidence")
    elif top_score < min_score:
        issues.append("no_evidence")
    if out_of_domain:
        issues.extend(["no_evidence", "out_of_domain"])
    elif documents and meaningful_query and overlap_ratio < MIN_KEYWORD_OVERLAP:
        issues.extend(["no_evidence", "low_keyword_overlap"])

    issues = list(dict.fromkeys(issues))
    return {
        "valid": not issues,
        "issues": issues,
        "top_score": round(top_score, 6),
        "min_score": min_score,
        "keyword_overlap": round(overlap_ratio, 3),
        "min_keyword_overlap": MIN_KEYWORD_OVERLAP,
        "overlap_tokens": overlap_tokens,
        "supported_terms": supported_terms,
        "out_of_domain_categories": matched_out_of_domain,
    }
