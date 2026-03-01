# [GPU Usage Policy]

### Default GPU Assignment

Unless explicitly specified otherwise, **always use GPU 2 only**.

### Execution Rule

All training / evaluation commands must set:

CUDA_VISIBLE_DEVICES=2

Example:

```bash
CUDA_VISIBLE_DEVICES=2 env -u VIRTUAL_ENV uv run python train.py ...
```
