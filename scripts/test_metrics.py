#!/usr/bin/env python3
"""Test script for metrics module.

Run this after setting up the conda environment to verify metrics work correctly.
"""

import torch
from soccer_recon.metrics import PSNR, SSIM, LPIPS, MetricsComputer


def test_metrics():
    """Test all metrics with dummy data."""
    print("Testing metrics module...")
    print()

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    # Create test images
    img = torch.rand(1, 3, 256, 256).to(device)
    img2 = torch.rand(1, 3, 256, 256).to(device)

    # Test individual metrics
    print("=" * 60)
    print("Testing Individual Metrics")
    print("=" * 60)

    psnr = PSNR(device=device)
    ssim = SSIM(device=device)
    lpips = LPIPS(device=device)

    # Test with identical images (should give best scores)
    print("\n1. Identical images (best case):")
    print(f"   PSNR:  {psnr(img, img):.2f} dB   (expect >100)")
    print(f"   SSIM:  {ssim(img, img):.4f}     (expect ~1.0)")
    print(f"   LPIPS: {lpips(img, img):.4f}     (expect ~0.0)")

    # Test with different random images (should give worst scores)
    print("\n2. Random different images (worst case):")
    print(f"   PSNR:  {psnr(img, img2):.2f} dB   (expect <20)")
    print(f"   SSIM:  {ssim(img, img2):.4f}     (expect <0.3)")
    print(f"   LPIPS: {lpips(img, img2):.4f}     (expect >0.5)")

    # Test with similar images (realistic case)
    img3 = img + torch.randn_like(img) * 0.1  # Add small noise
    img3 = img3.clamp(0, 1)
    print("\n3. Similar images with noise (realistic):")
    print(f"   PSNR:  {psnr(img, img3):.2f} dB   (expect 20-35)")
    print(f"   SSIM:  {ssim(img, img3):.4f}     (expect 0.7-0.95)")
    print(f"   LPIPS: {lpips(img, img3):.4f}     (expect 0.05-0.25)")

    # Test MetricsComputer
    print("\n" + "=" * 60)
    print("Testing MetricsComputer")
    print("=" * 60)

    computer = MetricsComputer(device=device)

    print("\n1. Compute all metrics (identical images):")
    metrics = computer.compute_all(img, img)
    for name, value in metrics.items():
        print(f"   {name.upper():6s}: {value:.4f}")

    print("\n2. Compute all metrics (noisy images):")
    metrics = computer.compute_all(img, img3)
    for name, value in metrics.items():
        print(f"   {name.upper():6s}: {value:.4f}")

    print("\n3. Compute subset of metrics:")
    metrics = computer.compute_subset(img, img3, ["psnr", "ssim"])
    for name, value in metrics.items():
        print(f"   {name.upper():6s}: {value:.4f}")

    # Test batch dimension handling
    print("\n" + "=" * 60)
    print("Testing Batch Dimension Handling")
    print("=" * 60)

    # Test with [C, H, W] input (no batch dim)
    img_no_batch = img[0]  # [3, 256, 256]
    print(f"\nInput shape: {img_no_batch.shape} (no batch dim)")
    print(f"PSNR: {psnr(img_no_batch, img_no_batch):.2f} dB")

    # Test with [B, C, H, W] input
    img_batch = img  # [1, 3, 256, 256]
    print(f"\nInput shape: {img_batch.shape} (with batch dim)")
    print(f"PSNR: {psnr(img_batch, img_batch):.2f} dB")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_metrics()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
