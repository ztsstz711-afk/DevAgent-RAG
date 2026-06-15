# OpenAI-Compatible LLM Answer 使用说明

## 默认行为

DevAgent-RAG 默认不依赖大模型 API。文档加载、TF-IDF / embedding / hybrid retrieval、错误解析、evidence gate、citation 和质量检查都可以在本地运行。默认回答由 deterministic template answer generator 生成，因此 demo 和测试不要求配置 API key。

当 `OPENAI_API_KEY` 不存在时，`python scripts/run_llm_demo.py` 不会报错退出，而是打印明确提示并自动 fallback 到 template answer mode。

## API 的作用

配置 API 后，项目可以通过 OpenAI-compatible Chat Completions 接口，把已经检索并通过 evidence gate 的文档片段整理为更自然的 grounded answer。

API 只用于最后的 answer generation：

```text
query -> TF-IDF / embedding / hybrid retrieval -> evidence gate -> LLM answer generation
```

API 不参与文档加载、索引构建或检索。即使启用 LLM Answer，检索仍由本地 TF-IDF、sentence-transformers embedding 或 hybrid retriever 完成。无有效 evidence 时，系统直接拒答，不调用 API。

## 环境变量

- `OPENAI_API_KEY`：API 凭证。
- `OPENAI_BASE_URL`：OpenAI-compatible API 根地址，默认 `https://api.openai.com/v1`。
- `OPENAI_MODEL`：模型名称，默认 `gpt-4o-mini`。

使用以下命令检查配置。该脚本只读取环境变量，不会发送网络请求：

```bash
python scripts/check_llm_config.py
```

配置完成后运行：

```bash
python scripts/run_llm_demo.py
```

## Windows PowerShell

仅对当前 PowerShell 会话生效：

```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENAI_MODEL="gpt-4o-mini"

python scripts/check_llm_config.py
python scripts/run_llm_demo.py
```

## Linux / macOS Bash

仅对当前 shell 会话生效：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"

python scripts/check_llm_config.py
python scripts/run_llm_demo.py
```

## OpenAI 示例

```text
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

## DeepSeek 示例

DeepSeek 提供 OpenAI-compatible 接口。模型名和地址应以服务商当前文档为准：

```text
OPENAI_API_KEY=your-deepseek-key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

## 其他 OpenAI-Compatible 服务

```text
OPENAI_API_KEY=your-provider-key
OPENAI_BASE_URL=https://your-provider.example/v1
OPENAI_MODEL=provider-model-name
```

服务需要兼容 `POST /chat/completions`，并返回 OpenAI 风格的 `choices[0].message.content`。

## 本地 vLLM Server 示例

假设本地 vLLM 已在 `http://localhost:8000` 提供 OpenAI-compatible API：

```text
OPENAI_API_KEY=local-placeholder
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_MODEL=your-served-model-name
```

本地服务即使不校验 key，当前客户端仍要求 `OPENAI_API_KEY` 非空才能进入 openai answer mode，可使用不敏感的占位值。

## 安全说明

- 不要将真实 API key 写入源码、README、测试或示例配置。
- 不要提交 `.env` 文件；项目 `.gitignore` 已忽略 `.env` 和 `.env.*`。
- 不要在截图、日志或 issue 中暴露完整 key。
- `check_llm_config.py` 只显示脱敏后的少量前后字符，不显示完整 key。
- API 请求失败时系统会记录原因并 fallback 到 template answer，不影响本地检索能力。
