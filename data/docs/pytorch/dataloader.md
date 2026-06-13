# PyTorch DataLoader

PyTorch DataLoader batches samples from a Dataset and supports shuffling, multiprocessing workers, and pinned memory. Use a custom collate function when samples require special batching.

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```
