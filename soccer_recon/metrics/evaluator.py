"""Shared metrics computation for training and evaluation."""

from typing import Dict, List

import torch

from .psnr import PSNR
from .ssim import SSIM
from .lpips import LPIPS


class MetricsComputer:
    """Computes multiple image quality metrics.

    Centralized class for computing PSNR, SSIM, and LPIPS metrics.
    Used in both training (base_gs_model.py) and evaluation (eval.py).
    """

    def __init__(self, device: torch.device | str = "cuda"):
        """Initialize all metrics.

        Args:
            device: Device to place metrics on
        """
        device_str = str(device)
        self.device = device_str
        self.psnr = PSNR(device=device_str)
        self.ssim = SSIM(device=device_str)
        self.lpips = LPIPS(device=device_str)

    def compute_all(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all metrics.

        Args:
            pred: Predicted image tensor
            gt: Ground truth image tensor

        Returns:
            Dictionary with metric names and values
        """
        with torch.no_grad():
            return {
                "psnr": self.psnr(pred, gt),
                "ssim": self.ssim(pred, gt),
                "lpips": self.lpips(pred, gt),
            }

    def compute_subset(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        metrics_list: List[str]
    ) -> Dict[str, float]:
        """Compute only specified metrics.

        Args:
            pred: Predicted image tensor
            gt: Ground truth image tensor
            metrics_list: List of metric names to compute
                         (e.g., ["psnr", "ssim"])

        Returns:
            Dictionary with requested metric names and values
        """
        with torch.no_grad():
            results = {}
            for metric_name in metrics_list:
                if metric_name == "psnr":
                    results["psnr"] = self.psnr(pred, gt)
                elif metric_name == "ssim":
                    results["ssim"] = self.ssim(pred, gt)
                elif metric_name == "lpips":
                    results["lpips"] = self.lpips(pred, gt)
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")
            return results
