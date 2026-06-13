# DevAgent-RAG 面试笔记

## 项目背景

AI 开发者遇到的问题通常分为两类：一类是“某个框架或 API 应该怎么用”，另一类是带有日志、异常类型和运行环境信息的报错诊断。通用搜索能返回大量页面，但信息来源分散，回答也不一定能说明依据。因此这个项目希望把文档检索、错误识别、代码示例查找和回答质量检查组织成一个可观察、可评测的 Agent 工作流。

项目第一目标不是替代官方文档或构建生产级知识平台，而是实现一个本地可运行、行为可验证的 Agentic RAG MVP：输入技术问题或错误日志，输出带 citation 的回答、质量报告和 tool trace。

## 为什么不是普通 RAG

普通 RAG 常见流程是“检索一次，然后生成回答”。DevAgent-RAG 在检索前后增加了任务判断和质量控制：

1. 先判断输入是普通问题还是错误日志。
2. 错误输入先提取错误类型和细节。
3. 检索相关文档，并单独查找代码片段。
4. 根据 evidence 生成带引用回答。
5. 检查 citation、回答长度与 no-evidence 状态。
6. 保存每个节点的执行信息，便于调试和评估。

因此核心价值不只是“多调用几个工具”，而是把不同任务的处理顺序、状态和失败行为显式化。

## LangGraph 工作流设计

主流程保持为：

```text
route_task
  -> parse_error（错误输入）
  -> retrieve_docs
  -> find_code_snippets
  -> generate_answer
  -> check_quality
```

状态由 `AgentState` 传递，包括 question、route、error_info、documents、code_snippets、answer、quality 和 tool_trace。LangGraph 负责条件路由与节点编排，节点内部调用小而明确的工具函数。这样做的好处是流程可读、节点可单测，也容易查看每一步的输入输出。

## Task Router

Router 使用轻量规则调用 `error_parser`。如果检测到 CUDA OOM、`ModuleNotFoundError`、`OPENAI_API_KEY` 缺失或 `RuntimeError`，路由为 `error`，先执行错误解析；否则路由为 `question`，直接进入文档检索。

当前 router 是确定性规则，优点是无需模型、结果稳定、便于测试；不足是覆盖范围有限，对复杂自然语言错误描述的泛化能力不如分类模型或 LLM router。

## 工具设计

- `doc_search_tool`：按 query 检索相关文档 chunk。
- `error_parser_tool`：识别错误类别、匹配文本和部分错误细节。
- `code_snippet_finder_tool`：从包含 fenced code block 的 chunk 中检索示例，并优先选择已命中文档的同源代码。
- `quality_check_tool`：检查 citation、回答完整性与 no-evidence 状态，输出 `passed`、`score` 和 `issues`。

工具调用写入 `data/output/tool_trace.jsonl`，可用于解释“系统调用了什么、返回了什么、为什么得到当前答案”。

## 无依据问题处理

检索器使用 `min_score` 阈值过滤结果。如果 top-k 为空或没有结果达到阈值，answer generator 不继续拼接看似合理的答案，而是返回：

```text
未在当前文档知识库中找到明确依据。
```

质量报告设置 `passed=false`，并加入 `no_evidence`。这把“系统不知道”变成显式、可测试的产品行为，也降低模板或 LLM 使用文档外知识补全答案的风险。

## 三种检索方式

### TF-IDF

TF-IDF 使用 sklearn `TfidfVectorizer`，适合关键词、错误名、类名和配置项的精确匹配。它速度快、结果稳定、无需模型下载，是默认 baseline；不足是对同义表达和语义改写不敏感。

### Embedding

Embedding retrieval 使用 `sentence-transformers/all-MiniLM-L6-v2` 将 query 与 chunk 编码为向量，再用 NumPy cosine similarity 排序。它更擅长语义相似问题，但需要额外模型文件，分数阈值也需要针对语料校准。

### Hybrid

Hybrid 同时取得 TF-IDF 与 embedding 结果，对两组分数分别做 min-max normalization，再计算：

```text
final_score = alpha * embedding_score + (1 - alpha) * tfidf_score
```

它希望同时保留错误关键词的精确匹配和自然语言改写的语义召回。相同 source/chunk 会去重。当前融合方式简单透明，但还没有做更复杂的 reranking 或按查询类型动态调整 alpha。

## 评估方法

- `Hit@1`：第一个结果是否来自预期产品文档。
- `Hit@3`：前三个结果是否包含预期产品文档。
- `retrieval_non_empty_rate`：查询是否至少返回一个超过阈值的结果。
- `citation_rate`：有依据问题的回答是否带 citation。
- `route_accuracy`：普通问题和错误问题是否进入预期路由。
- `quality_pass_rate`：质量检查通过的比例。正确 no-evidence 拒答按当前定义为 `passed=false`。
- `tool_success_rate`：每个 case 是否执行了该路由要求的完整工具集合。

`evaluate.py` 评估 Agent 全流程，`retrieval_eval.py` 对比三种检索模式。当前数据是内置 sample docs 和固定 case，因此高分只用于回归验证，不能外推为生产效果。

## 真实 LLM 的 Grounded Answer

可选 LLM Answer 通过 OpenAI-compatible Chat Completions API 工作。Prompt 中明确要求：

1. 只能使用提供的 evidence。
2. 每个关键结论携带原始 citation。
3. evidence 不足时输出固定拒答。
4. 不补充文档之外的信息。
5. 使用 Answer、Evidence、Next Steps 的清晰结构。

工程上还做了两层保护：无 evidence 时不调用 API；无 key 或 API 失败时回退确定性模板，并记录 fallback 原因。当前没有实现 LLM 输出后的 citation 归属校验，因此生产化时还应验证每个引用是否真的支持对应结论。

## 项目不足与后续方向

- sample docs 和评测集较小，缺少真实大型语料、噪声文档和时间变化数据。
- Router 与 error parser 依赖规则，错误类型覆盖有限。
- TF-IDF、embedding 阈值和 hybrid alpha 尚未通过更大验证集系统调参。
- 当前 embedding 索引适合小型本地数据，没有增量更新、批处理优化或向量数据库。
- 没有 cross-encoder reranker、query rewriting、multi-hop retrieval 或基于反馈的检索改进。
- LLM Answer 还可以增加结构化输出校验、citation entailment 检查和 token/cost 统计。
- 生产化还需要文档版本管理、权限隔离、可观测性、缓存、并发控制和安全评测。

这些限制也是项目展示时应主动说明的边界：当前成果是一个完整、可运行、可评测的 Agentic RAG 工程闭环，而不是生产级知识平台。
