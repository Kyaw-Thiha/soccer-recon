#!/usr/bin/env python3
"""
Training script for Soccer 3D Gaussian Splatting.

Usage:
    python scripts/train.py --data data/SoccerNet \\
        --match-path "england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea" \\
        --action-id 0

    # Or using the registered method
    ns-train soccer-gs --data data/SoccerNet \\
        soccernet-data \\
        --match-path "england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea"
"""

import tyro
from pathlib import Path
from typing import Optional

from nerfstudio.configs.base_config import LoggingConfig, MachineConfig, ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from soccer_recon.data.datamanager import SoccerDataManagerConfig
from soccer_recon.data.dataparsers import SoccerDataParserConfig
from soccer_recon.models.base_gs_model import SoccerGSModelConfig


def main(
    data: Path = Path("data/SoccerNet"),
    match_path: Optional[str] = None,
    action_id: Optional[str] = None,
    output_dir: Path = Path("outputs"),
    experiment_name: str = "soccer-recon",
    max_num_iterations: int = 30000,
    steps_per_save: int = 2000,
    steps_per_eval_image: int = 500,
    steps_per_eval_all_images: int = 5000,
    viewer_enabled: bool = True,
    load_checkpoint: Optional[Path] = None,
) -> None:
    """Train a Soccer 3D Gaussian Splatting model.

    Args:
        data: Path to SoccerNet dataset root
        match_path: Relative path to match (e.g., 'england_epl/2016-2017/...')
        action_id: Action ID to reconstruct (default: first action)
        output_dir: Directory to save outputs
        experiment_name: Name for this experiment
        max_num_iterations: Maximum training iterations
        steps_per_save: Save checkpoint every N steps
        steps_per_eval_image: Evaluate single image every N steps
        steps_per_eval_all_images: Evaluate all images every N steps
        viewer_enabled: Enable web viewer
        load_checkpoint: Path to checkpoint to resume from
    """

    # Build dataparser config with user parameters
    dataparser_config = SoccerDataParserConfig(
        data=data,
        match_path=match_path,
        action_id=action_id,
        scale_factor=1.0,
        orientation_method="up",
        center_method="poses",
        auto_scale_poses=True,
        downscale_factor=1,
    )

    # Build datamanager config
    datamanager_config = SoccerDataManagerConfig(
        dataparser=dataparser_config,
        train_num_rays_per_batch=4096,
        eval_num_rays_per_batch=4096,
        camera_optimizer=None,
    )

    # Build model config
    model_config = SoccerGSModelConfig(
        cull_alpha_thresh=0.005,
        use_green_background=True,
        use_field_bounds=True,
        field_length=105.0,
        field_width=68.0,
        warmup_length=500,
        output_depth_during_training=True,
        rasterize_mode="antialiased",
    )

    # Build viewer config
    viewer_config = ViewerConfig(
        num_rays_per_chunk=1 << 15,
        quit_on_train_completion=not viewer_enabled,
    )

    # Build pipeline config
    pipeline_config = VanillaPipelineConfig(
        datamanager=datamanager_config,
        model=model_config,
    )

    # Create trainer configuration
    trainer_config = TrainerConfig(
        method_name="soccer-gs",
        experiment_name=experiment_name,
        output_dir=output_dir,
        max_num_iterations=max_num_iterations,
        steps_per_save=steps_per_save,
        steps_per_eval_image=steps_per_eval_image,
        steps_per_eval_all_images=steps_per_eval_all_images,
        mixed_precision=True,
        pipeline=pipeline_config,
        viewer=viewer_config,
        vis="viewer" if viewer_enabled else "wandb",
        load_config=load_checkpoint,
        relative_model_dir=Path("nerfstudio_models"),
        load_dir=load_checkpoint.parent if load_checkpoint else None,
        load_step=None,
        load_checkpoint=load_checkpoint,
        machine=MachineConfig(),
        logging=LoggingConfig(),
    )

    # Print configuration
    print("\n" + "=" * 80)
    print("Soccer 3D Gaussian Splatting Training")
    print("=" * 80)
    print(f"Data: {data}")
    print(f"Match: {match_path or 'Auto-detect first match'}")
    print(f"Action ID: {action_id or 'Auto-detect first action'}")
    print(f"Output: {output_dir / experiment_name}")
    print(f"Max iterations: {max_num_iterations:,}")
    print(f"Viewer: {'Enabled' if viewer_enabled else 'Disabled'}")
    print("=" * 80 + "\n")

    # Launch training
    trainer = trainer_config.setup()
    trainer.setup()
    trainer.train()


def entrypoint():
    """Entrypoint for use with console_scripts."""
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
