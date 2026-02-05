"""
Soccer Gaussian Splatting Method Configuration.

Defines the complete pipeline for training on SoccerNet data.
"""

from pathlib import Path

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from soccer_recon.data.datamanager import SoccerDataManagerConfig
from soccer_recon.data.dataparsers import SoccerDataParserConfig
from soccer_recon.models.base_gs_model import SoccerGSModelConfig


# Pipeline configuration for soccer reconstruction
_pipeline_config = VanillaPipelineConfig(
    datamanager=SoccerDataManagerConfig(
        dataparser=SoccerDataParserConfig(
            data=Path("data/SoccerNet"),  # Default path, can be overridden
            scale_factor=1.0,
            orientation_method="up",
            center_method="poses",
            auto_scale_poses=True,
            downscale_factor=1,  # Use full resolution
        ),
        train_num_rays_per_batch=4096,
        eval_num_rays_per_batch=4096,
        camera_optimizer=None,  # Disable camera optimization initially
    ),
    model=SoccerGSModelConfig(
        # Gaussian Splatting parameters
        cull_alpha_thresh=0.005,
        # Soccer-specific settings
        use_green_background=True,
        use_field_bounds=True,
        field_length=105.0,
        field_width=68.0,
        # Optimization settings
        warmup_length=500,
        output_depth_during_training=True,
        rasterize_mode="antialiased",
    ),
)

SoccerGSConfig = MethodSpecification(
    config=_pipeline_config,  # type: ignore
    description="Soccer 3D Gaussian Splatting using SoccerNet multi-view data",
)
