from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DOCS = {
    "openai/embeddings.md": """# OpenAI Embeddings

Embeddings convert text into numeric vectors for semantic search, clustering, and retrieval. Create embeddings in batches and store each vector with stable source metadata.

```python
result = client.embeddings.create(model="text-embedding-3-small", input=["first", "second"])
vectors = [item.embedding for item in result.data]
```
""",
    "openai/rate_limits.md": """# OpenAI Rate Limits

OpenAI API rate limits protect service reliability. Handle HTTP 429 responses with exponential backoff and jitter. Keep requests within request-per-minute and token-per-minute limits, and inspect response headers for current limits.

## Python retry example

```python
import random
import time

for attempt in range(5):
    try:
        response = client.responses.create(model="gpt-4.1-mini", input="Hello")
        break
    except RateLimitError:
        time.sleep((2 ** attempt) + random.random())
```
""",
    "openai/structured_outputs.md": """# OpenAI Structured Outputs

Structured Outputs constrain model responses to a supplied JSON schema. Define required fields and parse the validated response instead of extracting values from free-form text.

```python
response = client.responses.parse(model="gpt-4.1-mini", input=prompt, text_format=ResultSchema)
```
""",
    "openai/authentication.mdx": """# OpenAI Authentication

Set the `OPENAI_API_KEY` environment variable before creating the client. Never commit API keys to source control.

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```
""",
    "langchain/retrieval.md": """# LangChain Retrieval

Retrievers accept a query and return relevant documents. For small offline corpora, TF-IDF is a useful deterministic baseline. Split documents into coherent chunks and retain source metadata for citations.

```python
docs = retriever.invoke("How does retrieval work?")
```
""",
    "langchain/tool_calling.md": """# LangChain Tool Calling

LangChain models can bind tools described by names, argument schemas, and functions. Execute requested tool calls, return tool results to the model, and preserve call identifiers.

```python
model_with_tools = model.bind_tools([search_docs])
message = model_with_tools.invoke("Search the docs")
```
""",
    "pytorch/dataloader.md": """# PyTorch DataLoader

PyTorch DataLoader batches samples from a Dataset and supports shuffling, multiprocessing workers, and pinned memory. Use a custom collate function when samples require special batching.

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```
""",
    "pytorch/cuda_memory.md": """# PyTorch CUDA Memory

CUDA out of memory means the requested allocation exceeded available GPU memory. Reduce batch size or sequence length, enable gradient accumulation or mixed precision, and release unused tensors. Use memory summaries to identify large allocations.

```python
import torch
print(torch.cuda.memory_summary())
torch.cuda.empty_cache()
```
""",
    "pytorch/runtime_errors.md": """# PyTorch Runtime Errors

A RuntimeError about dtype mismatch means operands use incompatible tensor data types. Inspect tensor dtype values and convert inputs or model parameters to one consistent dtype before the operation.

```python
inputs = inputs.to(dtype=model.dtype)
```
""",
    "huggingface/tokenizers.md": """# HuggingFace Tokenizers

AutoTokenizer loads the tokenizer associated with a pretrained model. Configure padding and truncation explicitly, and inspect input_ids and attention_mask before model inference.

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```
""",
    "huggingface/model_loading.txt": """HuggingFace Model Loading

Load models with from_pretrained. For constrained devices, use an appropriate dtype, device_map, or quantization. Confirm that required packages are installed when ModuleNotFoundError appears.

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
```
""",
    "vllm/serving.md": """# vLLM Serving

vLLM serves compatible language models with paged attention and an OpenAI-compatible HTTP server. Tune max model length and GPU memory utilization when the server reports a CUDA OOM.

```bash
vllm serve model-name --max-model-len 4096 --gpu-memory-utilization 0.85
```
""",
    "vllm/openai_server.md": """# vLLM OpenAI-Compatible Server

Start `vllm serve` to expose OpenAI-compatible chat and completion endpoints. Point an OpenAI client at the local base URL and use the served model name.

```python
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
```
""",
    "llamafactory/training.md": """# LLaMAFactory Training

LLaMAFactory supports supervised fine-tuning and parameter-efficient methods such as LoRA. For limited GPU memory, reduce per-device batch size, use gradient accumulation, bf16 or fp16, and a shorter cutoff length.

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
cutoff_len: 1024
bf16: true
```
""",
    "llamafactory/lora_sft_dataset.md": """# LLaMAFactory LoRA SFT Dataset Format

For LoRA supervised fine-tuning, register the dataset and provide instruction records. Alpaca-style data uses instruction, input, and output fields; conversational data can use role-based messages.

```json
{"instruction": "Summarize", "input": "Document text", "output": "Short summary"}
```
""",
}


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def prepare_sample_docs(root: Path = ROOT / "data" / "docs") -> int:
    for relative, content in DOCS.items():
        target = root / relative
        _atomic_write(target, content.strip() + "\n")
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# LangGraph Basics\n", "StateGraph connects typed state and deterministic nodes."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["graph = builder.compile()\n", "result = graph.invoke(state)"]},
        ],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }
    notebook_path = root / "langchain" / "langgraph_basics.ipynb"
    _atomic_write(notebook_path, json.dumps(notebook, indent=2))
    return len(DOCS) + 1


if __name__ == "__main__":
    print(f"Prepared {prepare_sample_docs()} sample documents in data/docs")
