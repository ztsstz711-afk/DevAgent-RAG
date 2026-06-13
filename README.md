# DevAgent-RAG

DevAgent-RAG 是一个面向 AI 开发文档检索与报错诊断的 Agentic RAG 项目。系统使用 LangGraph 编排任务路由、错误解析、文档检索、代码示例检索、回答生成与质量检查，并输出引用依据和 tool trace。

当前项目使用内置 sample docs，适合作为本地可运行的 RAG/Agent 工程示例，不依赖外部文档 API。

## 核心能力

- **LangGraph Agent workflow**：`route_task -> parse_error / retrieve_docs / find_code_snippets -> generate_answer -> check_quality`
- **任务路由**：区分普通技术问题与 CUDA OOM、`ModuleNotFoundError`、API key 缺失、`RuntimeError` 等错误输入。
- **三种检索模式**：TF-IDF baseline、sentence-transformers embedding、关键词与语义分数融合的 hybrid retrieval。
- **可追溯回答**：引用格式为 `[product | source | chunk_id]`，并将工具调用过程写入 JSONL trace。
- **无依据拒答**：检索结果为空或低于阈值时返回“未在当前文档知识库中找到明确依据。”，并记录 `no_evidence`。
- **可选 LLM Answer**：可调用 OpenAI-compatible Chat Completions API；无 key 或请求失败时自动回退模板回答。
- **内置评测**：输出任务路由、检索、引用、质量、工具成功率与 Hit@K 指标，并提供三种检索模式的对比报告。

## 安装

推荐在项目虚拟环境中安装依赖：

```bash
pip install -r requirements.txt
python scripts/prepare_sample_docs.py
python scripts/build_index.py
```

`sentence-transformers` 首次使用默认模型时可能需要下载模型文件。默认 TF-IDF 模式不会触发模型下载。

## 快速运行

```bash
python scripts/ask.py "How to handle OpenAI rate limits?"
python scripts/debug.py "CUDA out of memory"
python scripts/run_demo.py
python scripts/evaluate.py
python scripts/run_llm_demo.py
python scripts/retrieval_eval.py
python -m unittest discover -s tests -v
```

## 检索模式

在 `configs/default.yaml` 中配置：

```yaml
retrieval:
  mode: tfidf
  top_k: 3
  min_score: 0.05
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  hybrid_alpha: 0.5
```

- `tfidf`：默认 baseline，使用 sklearn `TfidfVectorizer`，无需下载模型。
- `embedding`：使用 sentence-transformers 和 NumPy cosine similarity 进行语义检索。
- `hybrid`：对 TF-IDF 与 embedding 分数做 min-max normalization，再按 `hybrid_alpha` 融合。

Embedding 依赖或模型不可用时，embedding/hybrid 请求会给出清晰原因并回退到 TF-IDF，默认主流程不受影响。`python scripts/retrieval_eval.py` 会请求并对比三种模式，同时记录实际 backend 和 fallback 状态。

## 可选 LLM Answer

默认 answer generator 为本地模板，不需要 API。运行以下命令可尝试 OpenAI-compatible LLM Answer：

```bash
python scripts/run_llm_demo.py
```

环境变量：

```text
OPENAI_API_KEY
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

LLM prompt 要求只能基于 evidence 回答、关键结论携带 citation，并在依据不足时拒答。未设置 `OPENAI_API_KEY` 或 API 请求失败时，系统自动回退 template，并在结果及 tool trace 中记录原因。不要将 API key 提交到代码仓库。

## 文档处理

- 支持 Markdown、MDX、TXT 和 Jupyter Notebook。
- 按 Markdown 标题与 fenced code block 切分 chunk。
- chunk 保留 source、section、路径、代码标记等 metadata。
- sample docs 覆盖 OpenAI、LangChain、PyTorch、HuggingFace、vLLM 和 LLaMAFactory。

## 导入真实公开文档

仓库默认使用 `data/docs/` 下的 sample docs，可完全离线运行。若要检索真实公开技术文档，请手动将官方仓库 clone 到 `external/`，例如：

```bash
git clone https://github.com/openai/openai-cookbook external/openai-cookbook
git clone https://github.com/langchain-ai/docs external/langchain-docs
git clone https://github.com/pytorch/tutorials external/pytorch-tutorials

python scripts/prepare_external_docs.py
python scripts/build_index.py
python scripts/run_demo.py
python scripts/evaluate.py
python scripts/retrieval_eval.py
```

导入器还识别 `external/vllm-docs` 和 `external/llamafactory`。它递归读取 Markdown、MDX、TXT 和 Notebook，跳过构建目录、资源目录、二进制文件及超大文件，并将标准化副本和 metadata manifest 写入 `data/docs_imported/`。`build_index.py` 会同时索引 sample docs 与 imported docs，并生成 `data/output/index_stats.json`。

`external/`、`data/docs_imported/` 和本地索引不建议提交 Git；官方仓库的许可证与内容使用条件应由使用者自行确认。评测问题仍是内置 sample cases，但导入文档后会基于 sample 与 external 的联合索引运行，指标可能变化，且始终不代表真实生产效果。

## 输出文件

- `data/output/demo_results.md`：默认模板回答 demo。
- `data/output/tool_trace.jsonl`：逐问题工具调用记录。
- `data/output/eval_report.json` / `.md`：Agent 与 RAG 综合评测。
- `data/output/retrieval_eval.json` / `.md`：TF-IDF、embedding、hybrid 对比。
- `data/output/llm_demo_results.md`：可选 LLM Answer demo。
- `data/output/index_stats.json`：本次索引的文档、chunk、source 与导入数量统计。

## 评测边界

当前评测仅使用项目内置 sample docs 和固定问题集。报告中的 100% Hit@K、route accuracy 或 citation rate，只说明系统在这个小型受控集合上的行为符合预期，不代表生产环境效果，也不代表已经覆盖真实文档规模、复杂噪声、权限控制或线上稳定性问题。

更完整的面试讲解见 `docs/interview_notes.md`，简历表述见 `docs/resume_bullets.md`。
