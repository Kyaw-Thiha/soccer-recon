# Training Scripts

## `train.py`

Main training script for Soccer 3D Gaussian Splatting.

### Quick Usage

```bash
# From project root
python scripts/train.py \
    --data ./data/SoccerNet \
    --match-path "england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea" \
    --action-id 0
```

Or use the installed command:

```bash
soccer-train \
    --data ./data/SoccerNet \
    --match-path "england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea" \
    --action-id 0
```

### Arguments

Run with `--help` to see all options:
```bash
python scripts/train.py --help
```

Key arguments:
- `--data`: Path to SoccerNet dataset
- `--match-path`: Match directory (relative to data path)
- `--action-id`: Which action to reconstruct
- `--max-num-iterations`: Training steps (default: 30000)
- `--experiment-name`: Name for this run
- `--viewer-enabled`: Launch web viewer (default: True)

### Output

Training outputs are saved to:
```
outputs/
└── {experiment-name}/
    └── soccer-gs/
        ├── config.yml           # Full config
        ├── dataparser_transforms.json
        ├── nerfstudio_models/  # Checkpoints
        └── ...
```

## See Also

- [Full Training Guide](../docs/TRAINING.md) - Comprehensive training documentation
- [Data Pipeline](../soccer_recon/data/README.md) - Data preparation and loading
