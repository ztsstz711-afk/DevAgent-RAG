# HuggingFace Tokenizers

AutoTokenizer loads the tokenizer associated with a pretrained model. Configure padding and truncation explicitly, and inspect input_ids and attention_mask before model inference.

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```
