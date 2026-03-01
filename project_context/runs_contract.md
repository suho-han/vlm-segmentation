# [Runs Output Contract]

`runs/{dataset}/{model}/{exp_id}/`

- `config.yaml`, `git_commit.txt`, `metrics.json`, `train_history.json`
- `ckpt/best.pt`, `ckpt/last.pt`
- `pred_vis/` (20 images)

## [Backbone Compare Automation — Structural Metrics Focus]

### Script\

Use `scripts/backbone_compare.py` to automatically aggregate `metrics.json` across runs and produce:

- `runs/backbone_compare/summary.md`
- `runs/backbone_compare/summary.csv`

### Command

```bash
CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV uv run python scripts/backbone_compare.py \
  --runs_dir runs \
  --out_dir runs/backbone_compare
```
