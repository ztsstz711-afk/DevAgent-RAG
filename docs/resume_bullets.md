# DevAgent-RAG 简历 Bullet

## 中文版本

- 基于 LangGraph 构建面向 AI 开发文档检索与报错诊断的 Agentic RAG，编排任务路由、错误解析、文档/代码检索、证据门控、回答生成与质量检查，并输出 citation 和 tool trace。
- 实现 Markdown、MDX、TXT、Jupyter Notebook 解析及外部公开文档导入，保留 source metadata 并生成索引统计；使用 LLaMA-Factory 真实文档完成 46 documents、487 chunks 的本地验证。
- 实现 sklearn TF-IDF、sentence-transformers embedding 与 hybrid retrieval，使用 NumPy cosine similarity、min-max normalization 和加权融合，并通过 Hit@K、valid evidence hit rate 等指标对比检索效果。
- 设计 evidence gate、domain guard 与 no-evidence refusal，解决真实 README 中弱相关 deploy/API chunk 导致 unsupported query 误答的问题，并以 quality report、unsupported refusal rate 和 JSONL tool trace 验证行为。

## English Version

- Built a LangGraph-based Agentic RAG workflow for AI documentation search and error diagnosis, orchestrating task routing, error parsing, document/code retrieval, evidence gating, answer generation, quality checks, citations, and tool traces.
- Implemented parsing and metadata-preserving import for Markdown, MDX, TXT, and Jupyter Notebook sources; validated the local pipeline with LLaMA-Factory documentation covering 46 documents and 487 chunks.
- Developed TF-IDF, sentence-transformers embedding, and hybrid retrieval using NumPy cosine similarity, min-max score normalization, and weighted fusion; compared retrieval quality with Hit@K and valid-evidence metrics.
- Added evidence gating, domain guards, and no-evidence refusal to prevent weakly related README chunks from answering unsupported queries, with behavior verified through quality reports, refusal metrics, and JSONL tool traces.
