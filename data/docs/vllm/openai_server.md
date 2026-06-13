# vLLM OpenAI-Compatible Server

Start `vllm serve` to expose OpenAI-compatible chat and completion endpoints. Point an OpenAI client at the local base URL and use the served model name.

```python
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
```
