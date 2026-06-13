# LangChain Tool Calling

LangChain models can bind tools described by names, argument schemas, and functions. Execute requested tool calls, return tool results to the model, and preserve call identifiers.

```python
model_with_tools = model.bind_tools([search_docs])
message = model_with_tools.invoke("Search the docs")
```
