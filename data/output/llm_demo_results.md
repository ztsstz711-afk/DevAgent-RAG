# DevAgent-RAG LLM Answer Demo

Graph backend: `langgraph`  
Retrieval mode: `tfidf`  
Retriever backend: `sklearn_tfidf`

## Case 1: How to handle OpenAI rate limits?

Answer backend: `template`  
Fallback reason: `OPENAI_API_KEY is not set; falling back to template answer mode.`  
Quality passed: `True`

## Answer

For: `How to handle OpenAI rate limits?`

### Recommended approach
- OpenAI Rate Limits OpenAI API rate limits protect service reliability. Handle HTTP 429 responses with exponential backoff and jitter. Keep requests within request-per-minute and token-per-minute limits, and inspect response headers for current limits [openai | rate_limits.md | chunk_001]
- Python retry example [openai | rate_limits.md | chunk_002]

### Example

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
[openai | rate_limits.md | chunk_003]

### Verification

Apply the smallest relevant change, rerun the failing command, and check the logs before increasing scope.

## Case 2: CUDA out of memory when training a PyTorch model

Answer backend: `template`  
Fallback reason: `OPENAI_API_KEY is not set; falling back to template answer mode.`  
Quality passed: `True`

## Answer

For: `CUDA out of memory when training a PyTorch model`

Detected issue: **cuda_oom**. Start by confirming the failing component and its runtime configuration.

### Recommended approach
- PyTorch CUDA Memory CUDA out of memory means the requested allocation exceeded available GPU memory. Reduce batch size or sequence length, enable gradient accumulation or mixed precision, and release unused tensors. Use memory summaries to identify large allocations [pytorch | cuda_memory.md | chunk_001]
- LLaMAFactory Training LLaMAFactory supports supervised fine-tuning and parameter-efficient methods such as LoRA. For limited GPU memory, reduce per-device batch size, use gradient accumulation, bf16 or fp16, and a shorter cutoff length [llamafactory | training.md | chunk_001]

### Example

```python
import torch
print(torch.cuda.memory_summary())
torch.cuda.empty_cache()
```
[pytorch | cuda_memory.md | chunk_002]

### Verification

Apply the smallest relevant change, rerun the failing command, and check the logs before increasing scope.

## Case 3: How to deploy Kubernetes on AWS?

Answer backend: `template`  
Fallback reason: `no_evidence`  
Quality passed: `False`

未在当前文档知识库中找到明确依据。
