"""
Soccer Gaussian Splatting Model.

Based on Nerfstudio's Splatfacto model, customized for multi-view soccer footage.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


@dataclass
class SoccerGSModelConfig(SplatfactoModelConfig):
    """Soccer Gaussian Splatting Model Configuration.

    Extends Splatfacto with soccer-specific optimizations.
    """

    _target: Type = field(default_factory=lambda: SoccerGSModel)

    # Soccer-specific parameters
    use_green_background: bool = True
    """Use green background color for soccer field scenes"""

    use_field_bounds: bool = True
    """Whether to use soccer field dimensions to constrain scene bounds"""

    field_length: float = 105.0
    """Soccer field length in meters (standard: 105m)"""

    field_width: float = 68.0
    """Soccer field width in meters (standard: 68m)"""


class SoccerGSModel(SplatfactoModel):
    """Soccer Gaussian Splatting Model.

    Extends Nerfstudio's Splatfacto model with soccer-specific features:
    - Field-aware scene bounds
    - Optimized for broadcast camera viewpoints
    - Support for multi-view correspondences
    """

    config: SoccerGSModelConfig  # type: ignore

    def populate_modules(self):
        """Set up the model components."""
        super().populate_modules()

        # Override background color for soccer field
        if self.config.use_green_background:
            # Typical grass green in RGB [0-1]
            self.background_color = torch.tensor([0.2, 0.5, 0.2])

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Return callbacks for training.

        Extends parent callbacks with soccer-specific logging.
        """
        callbacks = super().get_training_callbacks(training_callback_attributes)

        # Add custom callbacks here if needed
        # e.g., field boundary visualization, player tracking metrics, etc.

        return callbacks

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Generate outputs for the given camera.

        Args:
            camera: Camera to render from

        Returns:
            Dictionary of outputs including rgb, depth, etc.
        """
        outputs = super().get_outputs(camera)

        # Add any soccer-specific outputs here
        # e.g., field segmentation, player masks, etc.

        return outputs

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute metrics for logging.

        Args:
            outputs: Model outputs
            batch: Batch of data

        Returns:
            Dictionary of metrics
        """
        metrics = super().get_metrics_dict(outputs, batch)

        # Compute image quality metrics during eval or periodically during training
        if not self.training or self._should_compute_metrics():
            # Lazy initialization of metrics
            if not hasattr(self, '_metrics'):
                from soccer_recon.metrics import MetricsComputer
                self._metrics = MetricsComputer(device=self.device)
                self._step_counter = 0

            self._step_counter += 1

            # Get images in correct format [B, C, H, W]
            pred_rgb = outputs["rgb"]
            gt_rgb = batch["image"]
            pred_rgb = self._prepare_for_metrics(pred_rgb, batch)
            gt_rgb = self._prepare_for_metrics(gt_rgb, batch)

            # Compute metrics
            with torch.no_grad():
                metrics["psnr"] = self._metrics.psnr.compute(pred_rgb, gt_rgb)
                metrics["ssim"] = self._metrics.ssim.compute(pred_rgb, gt_rgb)

                # LPIPS is expensive - compute less frequently
                if self._step_counter % 2 == 0:
                    metrics["lpips"] = self._metrics.lpips.compute(pred_rgb, gt_rgb)

        return metrics

    def _should_compute_metrics(self) -> bool:
        """Compute metrics every 500 steps during training."""
        return self.step % 500 == 0

    def _prepare_for_metrics(self, image: torch.Tensor, batch) -> torch.Tensor:
        """Reshape image to [B, C, H, W] format for metrics.

        Args:
            image: Image tensor in various formats
            batch: Batch containing metadata

        Returns:
            Image tensor in [B, C, H, W] format
        """
        # Handle different image formats from Nerfstudio
        if image.ndim == 2:  # [H*W, 3]
            # Get image dimensions from batch
            h = batch.get("height", None)
            w = batch.get("width", None)
            if h is None or w is None:
                # Try to infer from image shape
                num_pixels = image.shape[0]
                h = int(num_pixels ** 0.5)
                w = num_pixels // h
            image = image.reshape(h, w, 3)

        if image.ndim == 3:  # [H, W, 3]
            image = image.permute(2, 0, 1).unsqueeze(0)  # → [1, 3, H, W]

        # Ensure values are in [0, 1]
        image = image.clamp(0, 1)
        return image

    @staticmethod
    def get_field_scene_box(
        field_length: float = 105.0,
        field_width: float = 68.0,
        height: float = 10.0,
    ) -> SceneBox:
        """Get scene box aligned with soccer field dimensions.

        Args:
            field_length: Length of field in meters
            field_width: Width of field in meters
            height: Height above field in meters

        Returns:
            SceneBox centered on field
        """
        # Center the field at origin
        aabb = torch.tensor(
            [
                [-field_length / 2, -field_width / 2, 0.0],
                [field_length / 2, field_width / 2, height],
            ],
            dtype=torch.float32,
        )
        return SceneBox(aabb=aabb)
