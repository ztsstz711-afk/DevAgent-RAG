# PyTorch Runtime Errors

A RuntimeError about dtype mismatch means operands use incompatible tensor data types. Inspect tensor dtype values and convert inputs or model parameters to one consistent dtype before the operation.

```python
inputs = inputs.to(dtype=model.dtype)
```
