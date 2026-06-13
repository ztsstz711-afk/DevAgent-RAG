# LangChain Retrieval

Retrievers accept a query and return relevant documents. For small offline corpora, TF-IDF is a useful deterministic baseline. Split documents into coherent chunks and retain source metadata for citations.

```python
docs = retriever.invoke("How does retrieval work?")
```
