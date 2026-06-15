# DevAgent-RAG

面向 AI 开发文档检索与报错诊断的本地 Agentic RAG 项目。系统使用 LangGraph 编排任务路由、错误解析、文档检索、代码示例检索、证据门控、回答生成与质量检查，并输出 citation 和 tool trace。

项目默认使用本地 template answer，不需要任何大模型 API，可直接复现测试、CLI demo、评估和 Streamlit 页面。OpenAI-compatible API 仅作为可选的最终回答生成能力，不参与文档检索。

## Project Overview

DevAgent-RAG 接收技术问题或报错日志，从 OpenAI、LangChain/LangGraph、PyTorch、HuggingFace、vLLM、LLaMAFactory 等文档中检索依据，生成带引用的回答。系统支持内置 sample docs，也支持从本地 `external/` 目录导入真实公开文档仓库。

当前已使用 LLaMA-Factory 真实文档完成本地验证：

- `46` documents
- `487` chunks
- `30` imported documents
- `458` LLaMAFactory chunks

这些数字只描述当前本地验证语料，不代表生产规模或线上效果。

## Why This Project

AI 开发问题既包括 API/框架用法，也包括 CUDA OOM、依赖缺失、dtype mismatch 和 API key 配置等错误日志。单次“检索后生成”难以稳定处理不同任务，也容易把真实文档中的弱相关词误当成证据。

本项目重点验证以下工程问题：

- 如何用显式工作流区分技术问答与报错诊断。
- 如何比较关键词、语义和混合检索。
- 如何让回答携带可追溯 citation 和 tool trace。
- 如何在真实文档引入噪声后识别弱相关 evidence 并拒答。
- 如何在无 API key 时保持项目完整可运行。

## Key Features

- LangGraph Agentic RAG workflow。
- `doc_qa`、`error_debug`、`code_lookup`、`config_help` 展示层任务类型。
- TF-IDF、sentence-transformers embedding、hybrid retrieval。
- Markdown、MDX、TXT、Jupyter Notebook 加载与标题/代码块切分。
- 外部公开文档导入、metadata manifest 和索引统计。
- CUDA OOM、`ModuleNotFoundError`、API key 缺失、`RuntimeError` 解析。
- Evidence gate、domain guard 和 no-evidence refusal。
- Template answer 与可选 OpenAI-compatible LLM Answer。
- Citation、quality report、retrieved chunks 和 JSONL tool trace。
- CLI、Streamlit、本地评估和三模式检索对比报告。

## Architecture

```mermaid
flowchart LR
    Q["Question or error log"] --> R["route_task"]
    R -->|error| P["parse_error"]
    R -->|question| D["retrieve_docs"]
    P --> D
    D --> G["evidence gate / domain guard"]
    G --> C["find_code_snippets"]
    C --> A["template or optional LLM answer"]
    A --> QC["check_quality"]
    QC --> O["answer + citations + tool trace"]
```

核心流程：

```text
route_task -> parse_error / retrieve_docs -> evidence gate
           -> find_code_snippets -> generate_answer -> check_quality
```

API 只出现在 `generate_answer` 阶段。TF-IDF、embedding 和 hybrid retrieval 均在本地完成。

## Quick Start

```bash
python -m venv .venv
pip install -r requirements.txt
python scripts/prepare_sample_docs.py
python scripts/build_index.py
```

Windows PowerShell 可使用：

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run Tests

```bash
python -m unittest discover -s tests -v
```

当前回归测试覆盖 loader、splitter、三种 retrieval、Agent workflow、evidence gate、external import、LLM fallback、Web utils 和评估报告。测试通过只说明当前受控场景行为符合预期。

## Run Demo

```bash
python scripts/run_demo.py
python scripts/debug.py "CUDA out of memory"
python scripts/ask.py "How to handle OpenAI rate limits?"
python scripts/evaluate.py
python scripts/retrieval_eval.py
```

推荐完整演示流程见 [DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)。

## Streamlit Web Console

```bash
streamlit run app.py
```

本地控制台支持：

- 上传 `.md`、`.mdx`、`.txt`、`.ipynb`。
- 导入 GitHub 文档仓库并重建索引。
- 选择 retrieval mode、top-k 和 answer mode。
- 查看 task type、answer backend、quality report、citation、chunk 和 tool trace。
- 运行并预览 Agent eval 与 retrieval eval。

Web UI 是本地展示与操作入口，核心逻辑仍位于 `src/` 和 `scripts/`，未实现多用户权限、认证或公网部署安全能力。

## External Docs Import

默认 sample docs 可离线运行。真实文档不会提交仓库，需要手动 clone 到 `external/`：

```bash
git clone https://github.com/openai/openai-cookbook external/openai-cookbook
git clone https://github.com/langchain-ai/docs external/langchain-docs
git clone https://github.com/pytorch/tutorials external/pytorch-tutorials
git clone https://github.com/hiyouga/LLaMA-Factory external/llamafactory

python scripts/prepare_external_docs.py
python scripts/build_index.py
```

导入器限制文件类型、文件大小和每个 source 的数量，并跳过 `.git`、assets、build、dist、虚拟环境等目录。`external/`、`data/docs_imported/` 和 `data/index/` 已被 `.gitignore` 忽略。

## Retrieval Modes

在 `configs/default.yaml` 中配置：

```yaml
retrieval:
  mode: tfidf
  top_k: 3
  min_score: 0.05
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  hybrid_alpha: 0.5
```

| Mode | 说明 | 适用特点 |
|---|---|---|
| `tfidf` | sklearn TF-IDF baseline | 精确关键词、错误名、配置项，默认且无需模型下载 |
| `embedding` | sentence-transformers + NumPy cosine similarity | 语义改写与近义表达 |
| `hybrid` | min-max normalization 后融合两类分数 | 兼顾关键词匹配和语义召回 |

Embedding 模型不可用时会给出原因并回退 TF-IDF，不中断默认主流程。

## Optional LLM Answer

API 是可选能力。没有 `OPENAI_API_KEY` 时，demo 和测试仍可完整运行，`run_llm_demo.py` 会自动回退 template answer。

```bash
python scripts/check_llm_config.py
python scripts/run_llm_demo.py
```

环境变量：

```text
OPENAI_API_KEY
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

LLM 只能使用通过 evidence gate 的文档片段，并被要求保留 citation。无有效 evidence 时直接拒答，不调用 API。详细配置见 [API_USAGE.md](docs/API_USAGE.md)。

## Evaluation

```bash
python scripts/evaluate.py
python scripts/retrieval_eval.py
```

主要指标：

- Route accuracy
- Retrieval hit rate、Hit@1、Hit@3
- Retrieval non-empty rate 与 valid evidence hit rate
- Unsupported query refusal rate
- Citation rate、quality pass rate、tool success rate

输出文件：

- `data/output/eval_report.json` / `eval_report.md`
- `data/output/retrieval_eval.json` / `retrieval_eval.md`
- `data/output/demo_results.md`
- `data/output/llm_demo_results.md`
- `data/output/tool_trace.jsonl`
- `data/output/index_stats.json`

评测使用固定 sample cases。即使某些指标达到 100%，也只代表当前小型受控评测集，不代表真实生产质量。

## Project Stages

项目从 V1 的 LangGraph + TF-IDF MVP，逐步增加拒答与评估、可选 LLM、embedding/hybrid、真实文档导入、Streamlit 和 evidence gate。完整记录见 [PROJECT_STAGES.md](docs/PROJECT_STAGES.md)。

## Limitations

- 不是生产级知识平台，未实现账号、权限、审计和在线服务治理。
- External docs 和评测集规模有限，缺少持续更新与版本一致性管理。
- Router/error parser 主要依赖确定性规则，覆盖范围有限。
- Evidence gate 使用关键词类别和阈值，可能产生误拒答或漏拒答。
- Hybrid alpha、检索阈值尚未基于大型标注集调优。
- LLM 输出尚未实现严格的 citation entailment 验证与成本统计。

## Future Work

- 基于更大真实问答集调优阈值与 hybrid 权重。
- 增加 reranker、query rewriting 和 citation entailment 检查。
- 支持增量索引、文档版本管理和更稳定的 embedding cache。
- 完善可观测性、性能测试、权限隔离与安全评测。
- 将规则 router 与 LLM/classifier router 做可解释对比实验。

## Project Materials

- [面试话术](docs/INTERVIEW_NOTES.md)
- [中英文简历 Bullet](docs/RESUME_BULLETS.md)
- [演示脚本](docs/DEMO_SCRIPT.md)
- [API 配置说明](docs/API_USAGE.md)
- [项目阶段](docs/PROJECT_STAGES.md)
