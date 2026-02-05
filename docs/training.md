## Training Soccer 3D Gaussian Splatting

This is the minimal guide to run a training job.

## Quick Start

```bash
# 1) Install
pip install -e .

# 2) Download SoccerNet data
python soccer_recon/data/download_match.py --split train

# 3) Train one action
soccer-train \
  --data ./data/SoccerNet \
  --match-path "england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea" \
  --action-id 0 \
  --experiment-name leicester-chelsea-action0
```

## Nerfstudio CLI (equivalent)

```bash
ns-train soccer-gs \
  --data ./data/SoccerNet \
  soccernet-data \
  --match-path "england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea" \
  --action-id 0 \
  --experiment-name leicester-chelsea-action0
```

## Common args
- `--data`: dataset root
- `--match-path`: relative match directory
- `--action-id`: action index
- `--max-num-iterations`: training steps
- `--output-dir`: output root
- `--experiment-name`: run name
- `--viewer-enabled`: web viewer
- `--load-checkpoint`: resume from checkpoint

## View results
- Web viewer URL appears in the terminal (usually `http://localhost:7007`).
- Render a camera path:

```bash
ns-render camera-path \
  --load-config outputs/<run>/soccer-gs/config.yml \
  --camera-path-filename camera_path.json \
  --output-path renders/video.mp4
```

## Troubleshooting (short)
- OOM: reduce `train_num_rays_per_batch` or downscale images.
- Poor recon: check camera poses and use multiple views.
- Slow: reduce eval frequency or image size.
