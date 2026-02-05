"""
Soccer Data Manager for Nerfstudio.

Manages data loading and batching for soccer reconstruction.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.utils.dataloaders import CacheDataloader


@dataclass
class SoccerDataManagerConfig(VanillaDataManagerConfig):
    """Soccer data manager configuration."""

    _target: Type = field(default_factory=lambda: SoccerDataManager)

    cache_images: Literal["cpu", "gpu"] = "cpu"
    """Where to cache images for faster loading"""

    cache_images_type: Literal["uint8", "float32"] = "uint8"
    """Image cache dtype"""


class SoccerDataManager(VanillaDataManager):
    """Soccer data manager.

    Handles loading and caching of SoccerNet multi-view images.
    """

    config: SoccerDataManagerConfig

    def setup_train(self):
        """Set up training dataset."""
        super().setup_train()

        # Additional setup for soccer-specific data
        # e.g., preload field masks, player annotations, etc.

    def setup_eval(self):
        """Set up evaluation dataset."""
        super().setup_eval()

    def next_train(self, step: int) -> Tuple[torch.Tensor, Dict]:
        """Get next training batch.

        Args:
            step: Current training step

        Returns:
            Tuple of (ray bundle, batch dict)
        """
        return super().next_train(step)

    def next_eval(self, step: int) -> Tuple[torch.Tensor, Dict]:
        """Get next evaluation batch.

        Args:
            step: Current evaluation step

        Returns:
            Tuple of (ray bundle, batch dict)
        """
        return super().next_eval(step)

    def next_eval_image(self, step: int) -> Tuple[int, torch.Tensor, Dict]:
        """Get next evaluation image.

        Args:
            step: Current evaluation step

        Returns:
            Tuple of (camera index, ray bundle, batch dict)
        """
        return super().next_eval_image(step)
