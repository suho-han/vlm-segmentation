# [Data]

### DATA_ROOT Priority
1. **CLI:** `--data_root /path`
2. **Env:** `export DATA_ROOT=/path`
3. **Config:** `configs/defaults.yaml` (`data_root: "./data"`)

### Auto-discovery
Folders under `DATA_ROOT` are matched by keywords:
- **OCTA500-6M:** `OCTA500_6M`, `OCTA500-6M`, `OCTA500`
- **DRIVE:** `DRIVE`, `drive`

### Structure
```
data/
├── OCTA500_6M/          # img: *.bmp, label: *.bmp
│   ├── train/ images/ + labels/
│   ├── val/   images/ + labels/
│   └── test/  images/ + labels/
└── DRIVE/               # img: *.tif, mask: *.gif
    ├── train/ images/ + masks/
    └── test/  images/ + masks/
```

| Dataset        | Path                                                  | Format                    | Split                              |
| -------------- | ----------------------------------------------------- | ------------------------- | ---------------------------------- |
| **OCTA500-6M** | `./data/OCTA500_6M/{train,val,test}/{images,labels}/` | `*.bmp`                   | 180 / 20 / 100                     |
| **DRIVE**      | `./data/DRIVE/{train,test}/{images,masks}/`           | img=`*.tif`, mask=`*.gif` | 20 / 20 (val=train 80/20, seed=42) |
