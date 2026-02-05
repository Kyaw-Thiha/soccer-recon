"""Base metric interface for image quality evaluation."""

from abc import ABC, abstractmethod
from typing import Union

import torch


class BaseMetric(ABC):
    """Abstract base class for image quality metrics."""

    def __init__(self, device: Union[str, torch.device] = "cuda"):
        """Initialize metric.

        Args:
            device: Device to place metric computation on ('cuda', 'cpu', or torch.device)
        """
        self.device = device

    @abstractmethod
    def compute(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute metric between predicted and ground truth images.

        Args:
            pred: Predicted image tensor [B, C, H, W] or [C, H, W]
            gt: Ground truth image tensor [B, C, H, W] or [C, H, W]

        Returns:
            Metric value as tensor (scalar or per-batch)
        """
        pass

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Convenience wrapper that returns scalar metric value.

        Args:
            pred: Predicted image tensor
            gt: Ground truth image tensor

        Returns:
            Metric value as Python float
        """
        with torch.no_grad():
            pred = self._ensure_batch_dim(pred)
            gt = self._ensure_batch_dim(gt)
            result = self.compute(pred, gt)
            return float(result.mean().item())

    def _ensure_batch_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has batch dimension [B, C, H, W].

        Args:
            tensor: Image tensor [C, H, W] or [B, C, H, W]

        Returns:
            Tensor with batch dimension [B, C, H, W]
        """
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor
