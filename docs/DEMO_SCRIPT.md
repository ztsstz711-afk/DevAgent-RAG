# DevAgent-RAG 演示脚本

## 本地启动前检查

1. 激活虚拟环境并确认依赖：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. 检查 LLM 配置。API 非必需：

```bash
python scripts/check_llm_config.py
```

3. 准备文档并构建索引：

```bash
python scripts/prepare_sample_docs.py
python scripts/prepare_external_docs.py
python scripts/build_index.py
```

4. 快速确认测试与报告：

```bash
python -m unittest discover -s tests -v
python scripts/evaluate.py
python scripts/retrieval_eval.py
```

## 命令行 Demo 流程

### 1. 展示基础 Agentic RAG

```bash
python scripts/run_demo.py
```

讲解点：LangGraph workflow、template answer、citation、quality report 和 tool trace。

### 2. 展示报错诊断

```bash
python scripts/debug.py "CUDA out of memory when training a PyTorch model"
```

讲解点：`error_debug`、CUDA OOM parser、PyTorch/LLaMAFactory evidence、代码片段和完整 trace。

### 3. 展示可选 LLM Answer

```bash
python scripts/run_llm_demo.py
```

无 key 时展示自动 template fallback；有 key 时展示 OpenAI-compatible grounded answer。说明 API 只负责最终 answer generation。

### 4. 展示评估

```bash
python scripts/evaluate.py
python scripts/retrieval_eval.py
```

讲解点：route accuracy、Hit@K、valid evidence、unsupported refusal、citation 和 tool success。

## Streamlit Demo 流程

启动：

```bash
streamlit run app.py
```

推荐顺序：

1. 在 Sidebar 选择 `tfidf`，展示默认 baseline 和 backend。
2. 进入 Knowledge Base，展示 index stats、真实 imported docs 数量和 chunks per source。
3. 切换 Ask / Debug，依次运行下方四个推荐问题。
4. 展开 Quality Report、Evidence Citations、Retrieved Chunks 和 Tool Trace。
5. 切换 `embedding` 与 `hybrid`，说明 effective mode 和 backend。
6. 进入 Evaluation，展示 eval report 与 retrieval comparison。

## 推荐演示问题

### How to configure LoRA training in LLaMAFactory?

期望展示：

- `task_type=doc_qa`
- LLaMAFactory citation
- LoRA/training 相关 retrieved chunks
- 当前 retrieval mode 与 retriever backend
- `quality_report.passed=true`

### How to prepare LLaMAFactory SFT dataset format?

期望展示：

- SFT dataset format 文档依据
- `[llamafactory | ... | chunk_xxx]` citation
- instruction/input/output 或消息格式相关 chunk
- tool trace 中的 doc search 与 code snippet search

### CUDA out of memory when training a PyTorch model

期望展示：

- `task_type=error_debug`
- `cuda_oom` error parser 结果
- PyTorch CUDA memory 与训练配置依据
- 减小 batch size、mixed precision 等文档内建议及 citation
- error parser、retrieval、answer、quality 的完整 tool trace

### How to deploy Kubernetes on AWS?

期望展示：

- 即使真实 LLaMAFactory README 检索到 deploy/API 弱相关 chunk，也不会生成答案
- 固定拒答：“未在当前文档知识库中找到明确依据。”
- `quality_report.passed=false`
- issues 包含 `no_evidence`、`out_of_domain`
- Retrieved Chunks 仍可查看，用于解释误检索来源
- Tool Trace 展示 evidence gate 如何区分“retrieval non-empty”和“valid evidence”

## 演示时的关键表述

- 默认不需要 API，因此项目可复现。
- API 只用于最终回答，不参与检索。
- External docs 不提交仓库，演示机本地导入。
- 高评测指标来自固定 sample cases，不代表生产效果。
- 项目的亮点是可观察的 Agent workflow、三种检索对比，以及真实文档噪声下的 evidence gate/refusal 闭环。
