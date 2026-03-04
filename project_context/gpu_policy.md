# [GPU Usage Policy]

### Default GPU Assignment

Unless explicitly specified otherwise, **use GPU 0 (available one)**.

### Execution Rule

All training / evaluation commands must set:

CUDA_VISIBLE_DEVICES=0

Example:

```bash
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python train.py ...
```
