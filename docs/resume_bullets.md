# 中文简历 Bullet

- 基于 LangGraph 构建面向 AI 开发文档检索与报错诊断的 Agentic RAG，编排任务路由、错误解析、文档与代码检索、回答生成和质量检查，并输出 citation 与 tool trace。
- 实现 sklearn TF-IDF、sentence-transformers embedding 与 hybrid retrieval 三种模式，使用 NumPy cosine similarity、分数归一化和加权融合完成本地检索及对比评测。
- 设计 no-evidence 拒答与质量检查机制，对低于检索阈值的问题返回明确拒答，并通过 route accuracy、citation rate、Hit@K、tool success rate 等指标进行回归验证。
- 实现可选 OpenAI-compatible LLM Answer，在 evidence 约束下生成带引用回答；无 API key 或请求失败时自动回退模板模式，保证项目可离线运行与测试。
