# DevAgent-RAG Demo Results

Graph backend: `langgraph`  
Retrieval mode: `tfidf`  
Retriever backend: `sklearn_tfidf`

## Case 1: How to handle OpenAI rate limits?

Route: `question`  
Quality: `1.00`

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

## Case 2: CUDA out of memory while training a model

Route: `error`  
Quality: `1.00`

## Answer

For: `CUDA out of memory while training a model`

Detected issue: **cuda_oom**. Start by confirming the failing component and its runtime configuration.

### Recommended approach
- PyTorch CUDA Memory CUDA out of memory means the requested allocation exceeded available GPU memory. Reduce batch size or sequence length, enable gradient accumulation or mixed precision, and release unused tensors. Use memory summaries to identify large allocations [pytorch | cuda_memory.md | chunk_001]

### Example

```python
import torch
print(torch.cuda.memory_summary())
torch.cuda.empty_cache()
```
[pytorch | cuda_memory.md | chunk_002]

### Verification

Apply the smallest relevant change, rerun the failing command, and check the logs before increasing scope.

## Case 3: ModuleNotFoundError: No module named 'transformers'

Route: `error`  
Quality: `1.00`

## Answer

For: `ModuleNotFoundError: No module named 'transformers'`

Detected issue: **module_not_found**. Start by confirming the failing component and its runtime configuration.

### Recommended approach
- 2. Install transformers from the main branch [llamafactory | README.md | chunk_065]

### Example

```python
git clone -b main https://github.com/huggingface/transformers.git

cd transformers

pip install .
```
[llamafactory | README.md | chunk_066]

### Verification

Apply the smallest relevant change, rerun the failing command, and check the logs before increasing scope.

## Case 4: How should I configure LLaMAFactory for a small GPU?

Route: `question`  
Quality: `1.00`

## Answer

For: `How should I configure LLaMAFactory for a small GPU?`

### Recommended approach
- Verify GPU Access After installation, verify that Docker can access your GPU: [llamafactory | README.md | chunk_010]
- HuggingFace Tokenizers AutoTokenizer loads the tokenizer associated with a pretrained model. Configure padding and truncation explicitly, and inspect input_ids and attention_mask before model inference [huggingface | tokenizers.md | chunk_001]
- If successful, you should see your GPU information displayed [llamafactory | README.md | chunk_012]

### Example

```python
llamafactory-cli webui
```
[llamafactory | README.md | chunk_073]

### Verification

Apply the smallest relevant change, rerun the failing command, and check the logs before increasing scope.
