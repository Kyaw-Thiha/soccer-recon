# SoccerNet Data Management Scripts

This directory contains scripts for downloading and processing SoccerNet-v3 dataset.

**All commands below should be run from the project root directory.**

## Scripts

### 1. `download_match.py`
Downloads frames and labels for a specific SoccerNet match.

**Important:** SoccerNet downloads all games in a split, then verifies your requested game exists.

**Basic Usage:**
```bash
# Download Leicester vs Chelsea match from train split (run from project root)
python soccer_recon/data/download_match.py --split train

# Download from different league/season
python soccer_recon/data/download_match.py \
    --league spain_laliga \
    --season 2015-2016 \
    --game "2016-04-02 - 17-00 Barcelona 1 - 2 Real Madrid" \
    --split valid

# Specify custom output directory
python soccer_recon/data/download_match.py \
    --split train \
    --output_dir ./data/SoccerNet
```

**Arguments:**
- `--output_dir`: Where to save data (default: `./data/SoccerNet` from project root)
- `--league`: League name (default: `england_epl`)
- `--season`: Season (default: `2016-2017`)
- `--game`: Match identifier (default: `2017-01-14 - 20-30 Leicester 0 - 3 Chelsea`)
- `--split`: Dataset split - **REQUIRED** (`train`/`valid`/`test`)
- `--files`: Files to download (default: `Labels-v3.json Frames-v3.zip`)

### 2. `extract_frames.py`
Extracts frames from zip file and visualizes annotations (simple standalone tool).

**Basic Usage:**
```bash
# Extract and preview frames (run from project root)
python soccer_recon/data/extract_frames.py \
    "data/SoccerNet/england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea"

# Just extract
python soccer_recon/data/extract_frames.py <match_dir> --extract

# Just preview with more samples
python soccer_recon/data/extract_frames.py <match_dir> --preview --num_samples 10

# Overwrite existing frames
python soccer_recon/data/extract_frames.py <match_dir> --overwrite
```

**Arguments:**
- `match_dir`: Path to match directory (positional, required)
- `--extract`: Extract frames from zip
- `--preview`: Preview frames with annotations
- `--num_samples`: Number of frames to preview (default: 5)
- `--overwrite`: Overwrite existing extracted frames
- `--all`: Extract and preview (default if no flags specified)

### 3. `dataloader.py`
PyTorch dataset loader for SoccerNet-v3/v3D with multi-view support.

**Basic Usage:**
```bash
# Test dataloader on train split
python soccer_recon/data/dataloader.py \
    --SoccerNet_path ./data/SoccerNet \
    --split train \
    --tiny 5

# Load from zipped images (faster, less disk space)
python soccer_recon/data/dataloader.py \
    --SoccerNet_path ./data/SoccerNet \
    --split train \
    --zipped_images

# Use SoccerNet-v3D format (with camera calibration)
python soccer_recon/data/dataloader.py \
    --SoccerNet_path ./data/SoccerNet \
    --split train \
    --use_v3d
```

**Arguments:**
- `--SoccerNet_path`: Path to SoccerNet dataset (required)
- `--split`: Dataset split (default: `all`)
- `--tiny`: Load only N games for testing
- `--resolution_width`: Image width (default: 1920)
- `--resolution_height`: Image height (default: 1080)
- `--preload_images`: Preload all images into RAM
- `--zipped_images`: Read directly from zip (recommended)
- `--use_v3d`: Use Labels-v3D.json format
- `--num_workers`: Dataloader workers (default: 4)

### 4. `visualize.py`
Official SoccerNet visualization tool with multi-view correspondence support.

**Basic Usage:**
```bash
# Visualize all actions in train split
python soccer_recon/data/visualize.py \
    --SoccerNet_path ./data/SoccerNet \
    --save_path ./data/visualizations \
    --split train \
    --tiny 5

# Visualize from zipped images
python soccer_recon/data/visualize.py \
    --SoccerNet_path ./data/SoccerNet \
    --save_path ./data/visualizations \
    --split train \
    --zipped_images

# Visualize SoccerNet-v3D data
python soccer_recon/data/visualize.py \
    --SoccerNet_path ./data/SoccerNet \
    --save_path ./data/visualizations \
    --split train \
    --use_v3d
```

**Arguments:** (Same as dataloader.py, plus `--save_path`)
- `--save_path`: Where to save visualized images (default: `./data/visualizations`)

**Output Files:**
- `action_XXXX_view_Y_original.png`: Original frame
- `action_XXXX_view_Y_bboxes.png`: Frame with bounding boxes
- `action_XXXX_view_Y_lines.png`: Frame with field lines
- `action_XXXX_links.png`: Multi-view correspondence visualization

## Complete Workflows

**Run these commands from the project root directory:**

### Option A: Simple Extraction & Preview

```bash
# Step 1: Download match data
python soccer_recon/data/download_match.py --split train

# Step 2: Quick extract and preview
python soccer_recon/data/extract_frames.py \
    "data/SoccerNet/england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea" \
    --num_samples 10
```

### Option B: Full Dataset Visualization (with Multi-view Links)

```bash
# Step 1: Download match data
python soccer_recon/data/download_match.py --split train

# Step 2: Visualize using official tool (shows multi-view correspondences)
python soccer_recon/data/visualize.py \
    --SoccerNet_path ./data/SoccerNet \
    --save_path ./data/visualizations \
    --split train \
    --zipped_images \
    --tiny 5
```

### Option C: SoccerNet-v3D Workflow (with Camera Calibration)

```bash
# Step 1: Download v3D data (must have Labels-v3D.json)
python soccer_recon/data/download_match.py \
    --split train \
    --files Labels-v3D.json Frames-v3.zip

# Step 2: Visualize with v3D support
python soccer_recon/data/visualize.py \
    --SoccerNet_path ./data/SoccerNet \
    --save_path ./data/visualizations_v3d \
    --split train \
    --use_v3d \
    --zipped_images
```

**Note:** You need to know which split (train/valid/test) contains your game. Check the [SoccerNet dataset documentation](https://github.com/SoccerNet/SoccerNet-v3).

## Output Structure

After running both scripts:

```
data/SoccerNet/
└── england_epl/
    └── 2016-2017/
        └── 2017-01-14 - 20-30 Leicester 0 - 3 Chelsea/
            ├── Frames-v3.zip          # Original zip file
            ├── Labels-v3.json         # Annotations
            ├── frames/                # Extracted frames
            │   ├── 0.png
            │   ├── 1.png
            │   ├── 1_1.png           # Replay frames
            │   └── ...
            └── visualizations/        # Annotated preview images
                ├── vis_0.png
                ├── vis_1.png
                └── ...
```

## SoccerNet-v3 vs SoccerNet-v3D

| Feature | v3 | v3D |
|---------|----|----|
| **Labels File** | `Labels-v3.json` | `Labels-v3D.json` |
| **2D Annotations** | ✓ Bboxes, lines, links | ✓ Same as v3 |
| **Camera Calibration** | ✗ | ✓ Pan/tilt/roll, focal length, position |
| **3D Ball Position** | ✗ | ✓ Triangulated from multi-view |
| **Calibrated Replays** | ✗ | ✓ Synchronized multi-view |
| **Use Case** | 2D scene understanding | 3D reconstruction, camera tracking |
| **Paper** | [SoccerNet-v3](https://github.com/SoccerNet/SoccerNet-v3/) | [SoccerNet-v3D (CVPR 2025)](https://github.com/mguti97/SoccerNet-v3D) |

**Key v3D Extensions:**
- Camera calibration parameters for each view
- 3D ball position annotations (4,051 images)
- Multi-view synchronization metadata
- Jaccard index quality metrics

## Requirements

Make sure these packages are installed:
```bash
# Basic requirements
pip install SoccerNet opencv-python tqdm numpy

# For dataloader and visualization
pip install torch torchvision
```

Or install in editable mode from project root:
```bash
pip install -e .
```

## Visualization Colors

The preview visualizations use the following color coding:
- **Blue**: Team left
- **Red**: Team right
- **Yellow**: Goalkeeper
- **Gray**: Referee
- **Green**: Other objects

## Notes

- Frame naming: `X.png` are action frames, `X_Y.png` are replay frames
- The Labels-v3.json contains annotations for all frames
- Visualizations show bounding boxes with class labels and jersey numbers
- Lines (field markings) are drawn in blue
