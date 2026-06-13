# LLaMAFactory LoRA SFT Dataset Format

For LoRA supervised fine-tuning, register the dataset and provide instruction records. Alpaca-style data uses instruction, input, and output fields; conversational data can use role-based messages.

```json
{"instruction": "Summarize", "input": "Document text", "output": "Short summary"}
```
