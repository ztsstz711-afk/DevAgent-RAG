# DevAgent-RAG Evaluation Report

- Graph backend: `langgraph`
- Retrieval mode: `tfidf`
- Retriever backend: `sklearn_tfidf`
- Total cases: 15
- Route accuracy: 100.0%
- Retrieval hit rate: 92.3%
- Citation rate: 100.0%
- Quality pass rate: 86.7%
- No-evidence count: 2
- Unsupported query count: 2
- Tool success rate: 100.0%
- Hit@1: 92.3%
- Hit@3: 92.3%

## Cases

| Question | Expected product | Route | Retrieval | Quality | No evidence |
|---|---|---|---:|---:|---:|
| How do I create OpenAI embeddings? | openai | question | True | True | False |
| How to handle OpenAI rate limits? | openai | question | True | True | False |
| How do OpenAI structured outputs enforce a schema? | openai | question | True | True | False |
| How does a LangChain retriever work? | langchain | question | True | True | False |
| How do I use LangChain tool calling? | langchain | question | True | True | False |
| How should I configure a PyTorch DataLoader? | pytorch | question | True | True | False |
| PyTorch CUDA out of memory | pytorch | error | True | True | False |
| How do I load a HuggingFace tokenizer? | huggingface | question | True | True | False |
| How do I start the vLLM OpenAI server? | vllm | question | True | True | False |
| What is the LLaMAFactory LoRA SFT dataset format? | llamafactory | question | True | True | False |
| ModuleNotFoundError: No module named transformers | huggingface | error | False | True | False |
| OPENAI_API_KEY missing | openai | error | True | True | False |
| RuntimeError: dtype mismatch | pytorch | error | True | True | False |
| How to deploy Kubernetes on AWS? | unsupported | question | False | False | True |
| How to price an insurance product? | unsupported | question | False | False | True |
