# [Model Constraints]

- **SwinUNETR** (monai 1.5.2): `img_size` % 32 == 0 → OCTA=384, DRIVE=512
  - `spatial_dims=2`, no `img_size` argument (changed in monai≥1.4)
- **UNet++**: `in_ch = nb_filter[i] * j + nb_filter[i+1]` (j skips + 1 upsample)
- **SwinUNETR VLM smoke test**: Must use exp_card config (not `--image_size` flag) since `defaults.yaml` `swinunetr.img_size=400` takes precedence over CLI.
