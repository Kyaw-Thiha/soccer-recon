#!/usr/bin/env python3
"""
Extract and preview frames from SoccerNet-v3 zip files.

This script extracts frames from Frames-v3.zip and can optionally visualize
the annotations (bounding boxes, lines) on sample frames.
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


def extract_frames(
    zip_path: Path,
    output_dir: Path,
    overwrite: bool = False
) -> int:
    """
    Extract all frames from a Frames-v3.zip file.

    Args:
        zip_path: Path to the Frames-v3.zip file
        output_dir: Directory where to extract frames
        overwrite: If True, overwrite existing files

    Returns:
        Number of frames extracted
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    if not overwrite and any(output_dir.glob("*.png")):
        existing_count = len(list(output_dir.glob("*.png")))
        print(f"Found {existing_count} existing frames in {output_dir}")
        print("Use --overwrite to re-extract")
        return existing_count

    print(f"Extracting frames from {zip_path.name}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = [m for m in zip_ref.namelist() if m.endswith('.png')]

        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, output_dir)

    frame_count = len(list(output_dir.glob("*.png")))
    print(f"✓ Extracted {frame_count} frames to {output_dir}")

    return frame_count


def load_labels(labels_path: Path) -> dict:
    """Load Labels-v3.json file."""
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with open(labels_path, 'r') as f:
        return json.load(f)


def draw_bbox(image: np.ndarray, bbox: dict, color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    Draw a bounding box on an image.

    Args:
        image: Image array
        bbox: Bounding box dict with keys 'x', 'y', 'w', 'h', 'class', 'jersey_number'
        color: BGR color tuple

    Returns:
        Image with bounding box drawn
    """
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

    # Draw rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Prepare label text
    label_parts = []
    if 'class' in bbox:
        label_parts.append(bbox['class'])
    if 'jersey_number' in bbox and bbox['jersey_number'] != -1:
        label_parts.append(f"#{bbox['jersey_number']}")

    label = " ".join(label_parts)

    # Draw label background
    if label:
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x, y - text_h - baseline - 5), (x + text_w, y), color, -1)
        cv2.putText(image, label, (x, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def draw_line(image: np.ndarray, line: dict, color: tuple = (255, 0, 0)) -> np.ndarray:
    """
    Draw a line on an image.

    Args:
        image: Image array
        line: Line dict with 'points' list and 'class'
        color: BGR color tuple

    Returns:
        Image with line drawn
    """
    points = line['points']

    # Draw line segments
    for i in range(0, len(points) - 1, 2):
        if i + 3 < len(points):
            pt1 = (int(points[i]), int(points[i + 1]))
            pt2 = (int(points[i + 2]), int(points[i + 3]))
            cv2.line(image, pt1, pt2, color, 2)

    return image


def visualize_frame(
    frame_path: Path,
    labels: dict,
    frame_id: str,
    output_path: Optional[Path] = None
) -> np.ndarray:
    """
    Visualize a frame with its annotations.

    Args:
        frame_path: Path to the frame image
        labels: Labels dictionary from Labels-v3.json
        frame_id: Frame identifier (e.g., "7" or "7_1")
        output_path: If provided, save the visualized image here

    Returns:
        Annotated image
    """
    # Load image
    image = cv2.imread(str(frame_path))
    if image is None:
        raise ValueError(f"Could not load image: {frame_path}")

    # Get annotations for this frame
    frame_data = labels.get(frame_id, {})

    # Draw bounding boxes
    bboxes = frame_data.get('bounding_boxes', [])
    for bbox in bboxes:
        # Color code by class
        team = bbox.get('class', 'unknown')
        if 'team left' in team.lower():
            color = (255, 0, 0)  # Blue
        elif 'team right' in team.lower():
            color = (0, 0, 255)  # Red
        elif 'goalkeeper' in team.lower():
            color = (0, 255, 255)  # Yellow
        elif 'referee' in team.lower():
            color = (128, 128, 128)  # Gray
        else:
            color = (0, 255, 0)  # Green

        image = draw_bbox(image, bbox, color)

    # Draw lines
    lines = frame_data.get('lines', [])
    for line in lines:
        image = draw_line(image, line)

    # Add frame info
    info_text = f"Frame: {frame_id} | Bboxes: {len(bboxes)} | Lines: {len(lines)}"
    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        print(f"Saved visualization to {output_path}")

    return image


def preview_frames(
    match_dir: Path,
    num_samples: int = 5,
    save_visualizations: bool = True
):
    """
    Preview sample frames with annotations.

    Args:
        match_dir: Directory containing the match data
        num_samples: Number of frames to preview
        save_visualizations: If True, save visualized frames
    """
    frames_dir = match_dir / "frames"
    labels_path = match_dir / "Labels-v3.json"

    if not frames_dir.exists():
        print(f"Frames directory not found: {frames_dir}")
        print("Please extract frames first.")
        return

    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        return

    # Load labels
    print(f"Loading labels from {labels_path.name}...")
    labels = load_labels(labels_path)

    # Get all frame files
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return

    print(f"Found {len(frame_files)} frames")

    # Sample frames
    step = max(1, len(frame_files) // num_samples)
    sample_frames = frame_files[::step][:num_samples]

    # Create visualization directory
    vis_dir = match_dir / "visualizations"
    if save_visualizations:
        vis_dir.mkdir(exist_ok=True)

    print(f"\nPreviewing {len(sample_frames)} frames...")

    for frame_path in sample_frames:
        frame_id = frame_path.stem  # filename without extension

        output_path = vis_dir / f"vis_{frame_path.name}" if save_visualizations else None

        try:
            visualize_frame(frame_path, labels, frame_id, output_path)
        except Exception as e:
            print(f"Error visualizing {frame_id}: {e}")

    if save_visualizations:
        print(f"\n✓ Visualizations saved to {vis_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and preview SoccerNet-v3 frames"
    )
    parser.add_argument(
        "match_dir",
        type=str,
        help="Path to match directory (e.g., data/SoccerNet/england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea)"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract frames from Frames-v3.zip"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview sample frames with annotations"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of frames to preview (default: 5)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted frames"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract and preview frames"
    )

    args = parser.parse_args()

    match_dir = Path(args.match_dir)

    if not match_dir.exists():
        print(f"Error: Match directory not found: {match_dir}")
        return

    # If --all flag, do both
    do_extract = args.extract or args.all
    do_preview = args.preview or args.all

    # If no flags specified, default to --all
    if not do_extract and not do_preview:
        do_extract = True
        do_preview = True

    # Extract frames
    if do_extract:
        zip_path = match_dir / "Frames-v3.zip"
        output_dir = match_dir / "frames"

        try:
            extract_frames(zip_path, output_dir, args.overwrite)
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return

    # Preview frames
    if do_preview:
        try:
            preview_frames(match_dir, args.num_samples, save_visualizations=True)
        except Exception as e:
            print(f"Error previewing frames: {e}")
            return

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
