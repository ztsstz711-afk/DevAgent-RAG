# vLLM Serving

vLLM serves compatible language models with paged attention and an OpenAI-compatible HTTP server. Tune max model length and GPU memory utilization when the server reports a CUDA OOM.

```bash
vllm serve model-name --max-model-len 4096 --gpu-memory-utilization 0.85
```
