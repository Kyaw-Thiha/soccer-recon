# Metrics & Evaluation

Short reference for reconstruction quality metrics and evaluation.

## What gets computed
- **PSNR**: pixel-level fidelity
- **SSIM**: structural similarity
- **LPIPS**: perceptual similarity (slowest)

## During training
Metrics are logged automatically by the model.
- PSNR/SSIM: frequent
- LPIPS: less frequent (expensive)

Implementation: `soccer_recon/models/base_gs_model.py`

## Standalone evaluation
Evaluate a trained run:

```bash
python scripts/eval.py \
  --load-config outputs/<run>/soccer-gs/config.yml \
  --output-path eval_results/<name> \
  --render-images
```

## Programmatic usage

```python
from soccer_recon.metrics import MetricsComputer

metrics = MetricsComputer(device="cuda")
results = metrics.compute(pred_rgb, gt_rgb)  # dict with psnr/ssim/lpips
```

Inputs should be float tensors in range [0, 1], shape `[B, C, H, W]` or `[C, H, W]`.

## Output files
- `metrics.json`: machine-readable summary
- `report.md`: human-readable report (if enabled)

## Tips
- LPIPS is slow; use a subset of views if needed.
- Unusually high/low scores often mean mismatched image ranges or shapes.
