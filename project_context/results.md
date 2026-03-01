# [All Results]

**Config:** epochs=1000 (max), early_stopping=val_loss (patience 50), AdamW lr=1e-4, Dice+BCE loss. DRIVE batch=2, OCTA batch=4.

*(Results are being re-generated with 1000 epochs and early stopping in `runs_repro/`)*

### DRIVE

| exp_id                 | Dice | IoU | hd95 | β0 err | β1 err | best_val |
| ---------------------- | ---- | --- | ---- | ------ | ------ | -------- |
| DRIVE-B0-SwinUNETR     | —    | —   | —    | —      | —      | —        |
| DRIVE-B1-UNetPP        | —    | —   | —    | —      | —      | —        |
| DRIVE-B2-nnUNet        | —    | —   | —    | —      | —      | —        |
| DRIVE-B3-SegFormer     | —    | —   | —    | —      | —      | —        |
| DRIVE-B4-TransUNet     | —    | —   | —    | —      | —      | —        |
| DRIVE-V0-SwinUNETR-VLM | —    | —   | —    | —      | —      | —        |
| DRIVE-V1-SwinUNETR-VLM | —    | —   | —    | —      | —      | —        |

### OCTA500-6M

| exp_id                  | Dice | IoU | hd95 | β0 err | β1 err | best_val |
| ----------------------- | ---- | --- | ---- | ------ | ------ | -------- |
| OCTA6M-B0-SwinUNETR     | —    | —   | —    | —      | —      | —        |
| OCTA6M-B1-UNetPP        | —    | —   | —    | —      | —      | —        |
| OCTA6M-B2-nnUNet        | —    | —   | —    | —      | —      | —        |
| OCTA6M-B3-SegFormer     | —    | —   | —    | —      | —      | —        |
| OCTA6M-B4-TransUNet     | —    | —   | —    | —      | —      | —        |
| OCTA6M-V0-SwinUNETR-VLM | —    | —   | —    | —      | —      | —        |
| OCTA6M-V1-SwinUNETR-VLM | —    | —   | —    | —      | —      | —        |
