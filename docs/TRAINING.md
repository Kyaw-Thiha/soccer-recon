## Training Soccer 3D Gaussian Splatting

This guide explains how to train the soccer reconstruction model on SoccerNet data.

## Quick Start

```bash
# 1. Install the package
pip install -e .

# 2. Download SoccerNet data (see soccer_recon/data/README.md)
python soccer_recon/data/download_match.py --split train

# 3. Train on a specific action
soccer-train \
    --data ./data/SoccerNet \
    --match-path "england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea" \
    --action-id 0 \
    --experiment-name leicester-chelsea-action0
```

Current Training
```bash
python scripts/train.py --data ./data/SoccerNet --match-path "england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace" --action-id 5.png --experiment-name chelsea-palace-action5 --max-num-iterations 30000 --viewer-enabled
```

## Using Nerfstudio CLI

Since the method is registered with Nerfstudio, you can also use the standard `ns-train` command:

```bash
ns-train soccer-gs \
    --data ./data/SoccerNet \
    soccernet-data \
    --match-path "england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea" \
    --action-id 0 \
    --experiment-name leicester-chelsea-action0
```

## Training Arguments

### Data Arguments
- `--data`: Path to SoccerNet dataset root (default: `data/SoccerNet`)
- `--match-path`: Relative path to match directory
- `--action-id`: Action ID to reconstruct (auto-detects first action if not specified)

### Training Arguments
- `--max-num-iterations`: Maximum training steps (default: 30000)
- `--steps-per-save`: Save checkpoint every N steps (default: 2000)
- `--steps-per-eval-image`: Evaluate single image every N steps (default: 500)
- `--steps-per-eval-all-images`: Evaluate all images every N steps (default: 5000)

### Output Arguments
- `--output-dir`: Output directory (default: `outputs`)
- `--experiment-name`: Experiment name (default: `soccer-recon`)
- `--viewer-enabled`: Enable web viewer (default: True)

### Resume Training
- `--load-checkpoint`: Path to checkpoint to resume from

## Model Architecture

The soccer reconstruction pipeline consists of:

1. **SoccerDataParser** (`soccer_recon/data/dataparsers.py`)
   - Parses SoccerNet-v3D format
   - Extracts camera calibration
   - Handles multi-view correspondences

2. **SoccerDataManager** (`soccer_recon/data/datamanager.py`)
   - Manages data loading and caching
   - Batches rays for training

3. **SoccerGSModel** (`soccer_recon/models/base_gs_model.py`)
   - Based on Nerfstudio's Splatfacto
   - Soccer field-aware scene bounds
   - Green background for grass
   - Optimized for broadcast cameras

## Viewing Results

### Web Viewer
If viewer is enabled (default), open your browser to the URL shown in the terminal (usually `http://localhost:7007`).

### Rendering Videos
After training, render a camera path:

```bash
ns-render camera-path \
    --load-config outputs/leicester-chelsea-action0/soccer-gs/config.yml \
    --camera-path-filename camera_path.json \
    --output-path renders/video.mp4
```

### Export Point Cloud
Export the Gaussian splats:

```bash
ns-export pointcloud \
    --load-config outputs/leicester-chelsea-action0/soccer-gs/config.yml \
    --output-dir exports/pointcloud/ \
    --num-points 1000000
```

## Training Tips

### For Best Results
1. **Use SoccerNet-v3D data** - Camera calibration significantly improves reconstruction
2. **Multiple views** - More replay angles = better 3D reconstruction
3. **Field visibility** - Actions with clear field markings work best
4. **Adjust iterations** - Complex scenes may need 50k+ iterations

### Common Issues

**Out of Memory**
- Reduce `train_num_rays_per_batch` in method config
- Lower image resolution with `--downscale-factor 2`

**Poor Reconstruction**
- Check camera poses are correct (view in web viewer)
- Ensure action has multiple replay angles
- Try different `action-id` values

**Training Too Slow**
- Enable mixed precision (default: enabled)
- Reduce evaluation frequency
- Use smaller images

## Advanced Configuration

Edit `soccer_recon/methods/base_gs_method.py` to customize:
- Gaussian splatting parameters
- Scene bounds (field dimensions)
- Background color
- Optimization schedules

## Dataset Format

Expected directory structure:
```
data/SoccerNet/
└── england_epl/
    └── 2016-2017/
        └── 2017-01-14 - 20-30 Leicester 0 - 3 Chelsea/
            ├── Labels-v3D.json  # Camera calibration & annotations
            ├── Frames-v3.zip    # Multi-view images
            └── frames/          # Extracted frames (optional)
                ├── 0.png       # Action frame
                ├── 0_1.png     # Replay 1
                ├── 0_2.png     # Replay 2
                └── ...
```

## Monitoring Training

### Weights & Biases
Set up W&B integration:
```bash
wandb login
# Then train with --vis wandb
```

### TensorBoard
View logs:
```bash
tensorboard --logdir outputs/
```

## Next Steps

- Train on multiple actions to build a full match reconstruction
- Fine-tune for player tracking
- Integrate with ball trajectory estimation
- Export for game analysis tools
