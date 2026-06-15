# DevAgent-RAG 项目阶段

## V1: LangGraph + TF-IDF + Tools

建立 LangGraph 工作流，实现任务路由、文档检索、错误解析、代码片段查找、模板回答、质量检查和 tool trace，并使用本地 TF-IDF 作为 baseline。

## V1.5: No-Evidence Refusal + Evaluation

增加检索阈值、无依据拒答、citation quality check，以及 route accuracy、Hit@K、citation rate 和 tool success rate 等内置评测。

## V2.1: Optional LLM Answer

增加 OpenAI-compatible Chat Completions answer generator。默认仍使用模板；无 API key 或请求失败时自动 fallback。

## V2.2: Embedding + Hybrid Retrieval

加入 sentence-transformers embedding retrieval 与 TF-IDF/embedding 分数融合的 hybrid retrieval，并生成三种模式的对比报告。

## V2.3: External Docs Import

支持从 `external/` 导入真实公开 AI 技术文档，保留 metadata，联合 sample docs 构建索引并输出 index stats。

## V2.4: Streamlit Web Console

增加本地 Web 控制台，支持上传文档、导入 GitHub 仓库、构建索引、运行 Agent 和查看评估报告，核心逻辑继续复用 `src/` 与 `scripts/`。

## V2.5: Evidence Gate + Domain Guard

针对真实文档中的弱相关检索增加 evidence gate 和领域判断，区分“检索非空”与“证据有效”，防止 Kubernetes、保险等无关问题被误答。

## V2.6: API Config + Validation

补充 OpenAI-compatible API 配置文档和离线配置检查脚本，明确 API 只用于最终回答生成，不参与检索，并验证无 key fallback 行为。
