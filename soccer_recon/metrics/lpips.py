"""Learned Perceptual Image Patch Similarity (LPIPS) metric."""

import torch
import lpips as lpips_lib

from .base import BaseMetric


class LPIPS(BaseMetric):
    """Learned Perceptual Image Patch Similarity.

    Deep learning-based perceptual similarity metric.
    Range: [0, 1] where 0.0 indicates identical images (distance metric).
    Typical range: 0.05-0.25 for neural rendering.
    Note: Lower is better (this is a distance, not similarity).
    """

    def __init__(self, device: torch.device | str = "cuda", net: str = "alex"):
        """Initialize LPIPS metric.

        Args:
            device: Device to place computation on
            net: Network backbone ('alex', 'vgg', or 'squeeze')
                 'alex' is fastest and accurate
        """
        device_str = str(device)
        super().__init__(device_str)
        self.metric = lpips_lib.LPIPS(net=net).to(device_str)
        self.metric.eval()

    def compute(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS between predicted and ground truth images.

        Args:
            pred: Predicted image [B, C, H, W] in range [0, 1]
            gt: Ground truth image [B, C, H, W] in range [0, 1]

        Returns:
            LPIPS distance value (lower is better)
        """
        pred = self._ensure_batch_dim(pred).to(self.device)
        gt = self._ensure_batch_dim(gt).to(self.device)

        # Convert from [0, 1] to [-1, 1] as required by LPIPS
        pred = pred * 2.0 - 1.0
        gt = gt * 2.0 - 1.0

        # LPIPS returns [B, 1, 1, 1], squeeze to [B]
        result = self.metric(pred, gt)
        return result.squeeze()
