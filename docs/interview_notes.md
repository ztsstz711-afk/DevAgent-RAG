# DevAgent-RAG 面试话术

## 30 秒项目介绍

我做了一个面向 AI 开发文档检索和报错诊断的 Agentic RAG。它用 LangGraph 编排任务路由、错误解析、TF-IDF/embedding/hybrid 检索、代码示例查找、evidence gate、回答生成和质量检查。项目默认不需要大模型 API，也支持可选的 OpenAI-compatible LLM Answer。我还导入了 LLaMA-Factory 真实文档进行验证，并针对真实 README 中弱相关 chunk 导致误答的问题增加了 domain guard 和 no-evidence refusal。

## 1 分钟项目介绍

这个项目解决的是 AI 开发者查文档和排查报错时，普通 RAG 容易把所有输入都按同一条链路处理、且难以解释依据的问题。我用 LangGraph 把流程拆成 task routing、error parsing、document retrieval、code snippet search、evidence validation、answer generation 和 quality check。检索层支持 TF-IDF、sentence-transformers embedding 和 hybrid 三种模式；回答必须带 `[product | source | chunk_id]` citation，并输出 tool trace。

项目默认使用模板回答，所以没有 API key 也能跑测试、CLI、评估和 Streamlit。API 只用于最后的 grounded answer generation，不参与检索。我还支持从 external 目录导入真实公开文档，本地用 LLaMA-Factory 验证到 46 份文档、487 个 chunk。真实文档引入后曾出现 Kubernetes/AWS 问题命中 README 中 deploy/API 词而被误答，我通过 evidence gate、关键词重叠和 domain guard 把“检索非空”和“证据有效”区分开。

## 这和普通 RAG 有什么区别？

普通 RAG 常见流程是 query、retrieve、generate。这个项目多了三层控制：

1. 检索前有任务路由，错误日志先做 error parsing。
2. 检索后有 evidence gate 和 domain guard，不是检索到 chunk 就允许回答。
3. 回答后有 quality check，并记录 citation、拒答原因和 tool trace。

因此重点不是工具数量，而是把任务分流、证据有效性和失败行为显式化、可测试化。

## 为什么不用纯 LangChain？

LangChain 很适合提供 loader、retriever、tool 等组件，但这个项目需要清晰表达条件路由和节点状态，例如错误问题先解析、普通问题直接检索，以及每一步如何写入 tool trace。LangGraph 更适合把这些状态转换和分支显式化。项目并不是排斥 LangChain，而是用 LangGraph 负责 orchestration，底层检索和工具保持独立、可单测。

## API 在项目里做什么？

API 只负责最后的 answer generation。query 先经过本地 TF-IDF、embedding 或 hybrid retrieval，再经过 evidence gate；只有有效 evidence 才会传给 OpenAI-compatible Chat Completions。Prompt 要求只能依据 evidence、关键结论带 citation、依据不足时拒答。API 不参与文档解析、建索引和检索。

## 没有 API 怎么跑起来？

默认就是无 API 模式。Template answer generator 根据检索到的 evidence 组织回答，因此测试、demo、评估和 Streamlit 都能本地运行。`run_llm_demo.py` 在没有 `OPENAI_API_KEY` 时不会崩溃，而是打印 fallback 原因并使用模板回答。`check_llm_config.py` 可以离线检查配置，且不会发送网络请求。

## 真实文档导入后遇到了什么问题？

导入 LLaMA-Factory README 后，`How to deploy Kubernetes on AWS?` 会命中文档中的 Docker、deploy、OpenAI-style API 等通用词。TF-IDF 分数超过基础阈值，旧逻辑只要有 citation 就会回答，导致 unsupported query 被误答。

我的修复不是针对一句测试文本硬编码，而是增加 evidence gate：检查 top score、query/evidence 关键词重叠、支持领域关键词和无关领域类别。原始 retrieved chunks 仍保留给 UI 和 trace，但无效 evidence 不进入 answer generator。这样既能解释为什么拒答，也能保留调试信息。

## 怎么评估 RAG 效果？

我把评估拆成流程与检索两部分：

- Route accuracy：问题是否进入正确分支。
- Hit@1 / Hit@3：正确 source 是否出现在前 1/3 个结果。
- Retrieval non-empty rate：是否检索到结果。
- Valid evidence hit rate：检索结果是否通过证据门控并命中预期 source。
- Unsupported refusal rate：无关问题是否被正确拒答。
- Citation rate：回答是否携带引用。
- Quality pass rate：回答是否通过质量检查。
- Tool success rate：所需节点是否完整执行。

需要强调，当前评测是固定 sample cases，主要用于回归验证，不能等同于生产效果。

## Embedding/Hybrid 比 TF-IDF 好在哪里？

TF-IDF 对错误名、类名、配置项和精确关键词很强，速度快且可解释，但对同义改写弱。Embedding 更能识别语义相似表达，但可能召回语义宽泛的结果，也需要模型和阈值校准。Hybrid 对两类分数归一化后加权，希望同时保留精确匹配与语义召回。项目通过同一套 case 比较三种模式，而不是预设 embedding 一定优于 TF-IDF。

## 这个项目有什么不足？

- 文档和标注评测集仍然较小，不代表真实生产分布。
- Router、error parser 和 domain guard 以规则为主，覆盖范围有限。
- Evidence gate 可能误拒答，需要更大数据集调参。
- 没有 cross-encoder reranker、增量索引和文档版本管理。
- LLM 回答没有做严格的 citation entailment 验证。
- Streamlit 是本地展示控制台，没有多用户认证、权限隔离和线上治理。

我的定位是：这是一个功能闭环、可运行、可评测的 Agentic RAG 工程项目，不是生产级知识平台。
