# OpenAI Embeddings

Embeddings convert text into numeric vectors for semantic search, clustering, and retrieval. Create embeddings in batches and store each vector with stable source metadata.

```python
result = client.embeddings.create(model="text-embedding-3-small", input=["first", "second"])
vectors = [item.embedding for item in result.data]
```
