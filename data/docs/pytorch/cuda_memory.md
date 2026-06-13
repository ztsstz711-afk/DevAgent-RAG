# PyTorch CUDA Memory

CUDA out of memory means the requested allocation exceeded available GPU memory. Reduce batch size or sequence length, enable gradient accumulation or mixed precision, and release unused tensors. Use memory summaries to identify large allocations.

```python
import torch
print(torch.cuda.memory_summary())
torch.cuda.empty_cache()
```
