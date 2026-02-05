"""Peak Signal-to-Noise Ratio (PSNR) metric."""

import torch
from torchmetrics.image import PeakSignalNoiseRatio

from .base import BaseMetric


class PSNR(BaseMetric):
    """Peak Signal-to-Noise Ratio metric.

    Measures reconstruction quality in decibels (dB).
    Higher values indicate better quality.
    Typical range: 20-35 dB for neural rendering.
    """

    def __init__(self, device: torch.device | str = "cuda", data_range: float = 1.0):
        """Initialize PSNR metric.

        Args:
            device: Device to place computation on
            data_range: Maximum value range of images (1.0 for normalized images)
        """
        super().__init__(device)
        self.metric = PeakSignalNoiseRatio(data_range=data_range).to(device)

    def compute(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute PSNR between predicted and ground truth images.

        Args:
            pred: Predicted image [B, C, H, W]
            gt: Ground truth image [B, C, H, W]

        Returns:
            PSNR value in dB
        """
        pred = self._ensure_batch_dim(pred).to(self.device)
        gt = self._ensure_batch_dim(gt).to(self.device)
        return self.metric(pred, gt)
