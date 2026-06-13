# LLaMAFactory Training

LLaMAFactory supports supervised fine-tuning and parameter-efficient methods such as LoRA. For limited GPU memory, reduce per-device batch size, use gradient accumulation, bf16 or fp16, and a shorter cutoff length.

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
cutoff_len: 1024
bf16: true
```
