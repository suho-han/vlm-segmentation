# [Metrics Keys (LOCK)]

Required keys in `metrics.json`: `Dice`, `IoU`, `hd95`, `betti_beta0`, `betti_beta1`

| Key | Description | Edge Case |
|----|------|------------|
| `Dice` | Sørensen–Dice | empty pred/gt → ~0 (Laplace) |
| `IoU` | Jaccard Index | empty pred/gt → ~0 |
| `hd95` | 95th Hausdorff (px) | empty pred or gt → **inf** |
| `betti_beta0` | \|β0_pred - β0_gt\| | connectivity (skimage label) |
| `betti_beta1` | \|β1_pred - β1_gt\| | holes (complement components) |

### Betti Proxy Implementation
- **β0:** `skimage.measure.label` 8-connectivity.
- **β1:** complement 1-connectivity components - 1.
- Euler characteristic based proxy (not full persistent homology).
