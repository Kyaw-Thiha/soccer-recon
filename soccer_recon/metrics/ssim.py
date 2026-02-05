"""Structural Similarity Index Measure (SSIM) metric."""

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from .base import BaseMetric


class SSIM(BaseMetric):
    """Structural Similarity Index Measure.

    Measures perceptual similarity between images.
    Range: [0, 1] where 1.0 indicates identical images.
    Typical range: 0.7-0.95 for neural rendering.
    """

    def __init__(self, device: torch.device | str = "cuda", kernel_size: int = 11):
        """Initialize SSIM metric.

        Args:
            device: Device to place computation on
            kernel_size: Size of Gaussian kernel (standard is 11)
        """
        super().__init__(device)
        self.metric = StructuralSimilarityIndexMeasure(
            kernel_size=kernel_size
        ).to(device)

    def compute(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between predicted and ground truth images.

        Args:
            pred: Predicted image [B, C, H, W]
            gt: Ground truth image [B, C, H, W]

        Returns:
            SSIM value in [0, 1]
        """
        pred = self._ensure_batch_dim(pred).to(self.device)
        gt = self._ensure_batch_dim(gt).to(self.device)
        return self.metric(pred, gt)
