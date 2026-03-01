# [Environment]

- **Path:** `/data1/suhohan/vlm-segmentation`
- **venv:** `.venv/` (Python 3.11, torch 2.10.0+cu128, monai 1.5.2, einops, open_clip_torch 3.3.0, transformers)
- **Package Management:** `uv` exclusive
- **Note:** `VIRTUAL_ENV=/data1/suhohan/JiT/.venv` is set in the shell.
  - → Always use `env -u VIRTUAL_ENV uv run ...`
  - → `uv pip install` requires `--python .venv/bin/python`

### Setup (uv)

```bash
uv venv                        # .venv 생성 (Python 3.11)
source .venv/bin/activate

# Install dependencies
uv pip install -e .
uv pip install monai einops    # SwinUNETR deps
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Verify
uv run pytest -q
```
