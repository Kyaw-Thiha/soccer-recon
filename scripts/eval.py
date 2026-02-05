#!/usr/bin/env python3
"""Standalone evaluation script for trained models.

Computes PSNR, SSIM, and LPIPS metrics on test views and generates reports.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import tyro
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def load_checkpoint_and_pipeline(config_path: Path):
    """Load trained checkpoint and pipeline.

    Args:
        config_path: Path to config.yml file

    Returns:
        Tuple of (config_dict, pipeline)
    """
    try:
        from nerfstudio.utils.eval_utils import eval_setup
        import yaml

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Set up evaluation
        _, pipeline, _, _ = eval_setup(
            config_path=config_path,
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )

        return config, pipeline

    except Exception as e:
        console.print(f"[red]Error loading checkpoint: {e}[/red]")
        raise


def prepare_image_for_metrics(
    image: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Convert image to [B, C, H, W] format for metrics.

    Args:
        image: Image tensor in various formats
        height: Target height
        width: Target width

    Returns:
        Image tensor in [1, C, H, W] format
    """
    if image.ndim == 2:  # [H*W, C]
        image = image.reshape(height, width, -1)

    if image.ndim == 3:  # [H, W, C]
        image = image.permute(2, 0, 1).unsqueeze(0)  # → [1, C, H, W]

    return image.clamp(0, 1)


def save_comparison_image(
    pred: torch.Tensor,
    gt: torch.Tensor,
    path: Path,
):
    """Save pred | gt | diff side-by-side comparison.

    Args:
        pred: Predicted image [1, C, H, W]
        gt: Ground truth image [1, C, H, W]
        path: Output path
    """
    # Convert to [H, W, C] and numpy
    pred_np = pred[0].permute(1, 2, 0).cpu().numpy()
    gt_np = gt[0].permute(1, 2, 0).cpu().numpy()

    # Compute absolute difference
    diff_np = np.abs(pred_np - gt_np)

    # Concatenate horizontally
    comparison = np.concatenate([pred_np, gt_np, diff_np * 3], axis=1)

    # Convert to uint8 and save
    comparison_uint8 = (comparison * 255).astype(np.uint8)
    Image.fromarray(comparison_uint8).save(path)


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict:
    """Compute statistics across all views.

    Args:
        all_metrics: List of per-view metric dictionaries

    Returns:
        Aggregated statistics with mean, std, min, max, per_view
    """
    result = {}

    # Get all metric names from first view
    metric_names = list(all_metrics[0].keys())

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        result[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "per_view": [float(v) for v in values],
        }

    return result


def save_json_report(summary: Dict, output_path: Path):
    """Save JSON report with detailed metrics.

    Args:
        summary: Aggregated metrics summary
        output_path: Output directory
    """
    report_path = output_path / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"[green]Saved JSON report to {report_path}[/green]")


def save_markdown_report(summary: Dict, output_path: Path):
    """Save markdown report with formatted tables.

    Args:
        summary: Aggregated metrics summary
        output_path: Output directory
    """
    report_path = output_path / "report.md"

    with open(report_path, "w") as f:
        # Header
        f.write("# Evaluation Report\n\n")
        f.write(f"**Checkpoint:** {summary['checkpoint']}\n")
        f.write(f"**Dataset:** {summary['dataset']}\n")
        f.write(f"**Views:** {summary['num_views']}\n")
        f.write(f"**Date:** {summary['timestamp']}\n\n")

        # Metrics summary table
        f.write("## Metrics Summary\n\n")
        f.write("| Metric | Mean  | Std  | Min   | Max   |\n")
        f.write("|--------|-------|------|-------|-------|\n")

        for metric_name, stats in summary["metrics"].items():
            f.write(
                f"| {metric_name.upper():6s} | "
                f"{stats['mean']:.3f} | "
                f"{stats['std']:.3f} | "
                f"{stats['min']:.3f} | "
                f"{stats['max']:.3f} |\n"
            )

        # Per-view breakdown
        f.write("\n## Per-View Breakdown\n\n")
        f.write("| View | PSNR | SSIM | LPIPS |\n")
        f.write("|------|------|------|-------|\n")

        num_views = len(summary["metrics"]["psnr"]["per_view"])
        for i in range(num_views):
            psnr = summary["metrics"]["psnr"]["per_view"][i]
            ssim = summary["metrics"]["ssim"]["per_view"][i]
            lpips = summary["metrics"]["lpips"]["per_view"][i]
            f.write(f"| {i:4d} | {psnr:.2f} | {ssim:.3f} | {lpips:.3f} |\n")

    console.print(f"[green]Saved markdown report to {report_path}[/green]")


def print_summary_table(summary: Dict):
    """Print summary table to console.

    Args:
        summary: Aggregated metrics summary
    """
    table = Table(title="Evaluation Results")

    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for metric_name, stats in summary["metrics"].items():
        table.add_row(
            metric_name.upper(),
            f"{stats['mean']:.3f}",
            f"{stats['std']:.3f}",
            f"{stats['min']:.3f}",
            f"{stats['max']:.3f}",
        )

    console.print(table)


def evaluate_checkpoint(
    load_config: Path,
    output_path: Path = Path("eval_results"),
    render_images: bool = True,
):
    """Evaluate trained checkpoint on test views.

    Args:
        load_config: Path to config.yml file
        output_path: Directory to save results
        render_images: Whether to save comparison images
    """
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and pipeline
    console.print("[cyan]Loading checkpoint...[/cyan]")
    config, pipeline = load_checkpoint_and_pipeline(load_config)

    # Initialize metrics
    from soccer_recon.metrics import MetricsComputer

    device = pipeline.device
    metrics_computer = MetricsComputer(device=str(device))

    # Get evaluation dataloader
    pipeline.eval()
    datamanager = pipeline.datamanager

    # Get test dataset
    eval_dataset = datamanager.eval_dataset
    if eval_dataset is None:
        raise RuntimeError("No evaluation dataset available in datamanager.")
    num_images = len(eval_dataset)

    console.print(f"[cyan]Evaluating on {num_images} test views...[/cyan]")

    # Evaluate each view
    all_metrics = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating views...", total=num_images)

        for img_idx in range(num_images):
            # Get camera and ground truth
            camera, batch = eval_dataset.get_data(img_idx)
            camera = camera.to(device)

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Render
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera(camera)

            # Get images
            height = camera.height[0].item()
            width = camera.width[0].item()

            pred_rgb = prepare_image_for_metrics(outputs["rgb"], height, width)
            gt_rgb = prepare_image_for_metrics(batch["image"], height, width)

            # Compute metrics
            view_metrics = metrics_computer.compute_all(pred_rgb, gt_rgb)
            all_metrics.append(view_metrics)

            # Save comparison image
            if render_images:
                img_path = output_path / f"view_{img_idx:03d}.png"
                save_comparison_image(pred_rgb, gt_rgb, img_path)

            progress.update(task, advance=1)

    # Aggregate results
    console.print("[cyan]Aggregating results...[/cyan]")
    aggregated = aggregate_metrics(all_metrics)

    # Create summary
    summary = {
        "checkpoint": str(load_config),
        "dataset": config.get("data", "unknown"),
        "num_views": num_images,
        "metrics": aggregated,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save reports
    save_json_report(summary, output_path)
    save_markdown_report(summary, output_path)

    # Print summary
    print_summary_table(summary)

    console.print("\n[green]✓ Evaluation complete![/green]")
    console.print(f"[green]Results saved to {output_path}[/green]")

    return summary


def main(
    load_config: Path,
    output_path: Path = Path("eval_results"),
    render_images: bool = True,
):
    """Main evaluation entry point.

    Args:
        load_config: Path to config.yml from training run
        output_path: Directory to save evaluation results
        render_images: Whether to render and save comparison images
    """
    if not load_config.exists():
        console.print(f"[red]Error: Config file not found at {load_config}[/red]")
        sys.exit(1)

    try:
        evaluate_checkpoint(
            load_config=load_config,
            output_path=output_path,
            render_images=render_images,
        )
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    tyro.cli(main)
