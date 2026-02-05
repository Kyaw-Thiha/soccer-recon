# SoccerNet Data Management Scripts

This directory contains scripts for downloading and processing SoccerNet-v3 dataset.

## Scripts

### 1. `download_match.py`
Downloads frames and labels for a specific SoccerNet match.

**Important:** SoccerNet downloads all games in a split, then verifies your requested game exists.

**Basic Usage:**
```bash
# Download Leicester vs Chelsea match from train split
python download_match.py --split train

# Download from different league/season
python download_match.py \
    --league spain_laliga \
    --season 2015-2016 \
    --game "2016-04-02 - 17-00 Barcelona 1 - 2 Real Madrid" \
    --split valid
```

**Arguments:**
- `--output_dir`: Where to save data (default: `../../data/SoccerNet`)
- `--league`: League name (default: `england_epl`)
- `--season`: Season (default: `2016-2017`)
- `--game`: Match identifier (default: Leicester vs Chelsea)
- `--split`: Dataset split - **REQUIRED** (`train`/`valid`/`test`)
- `--files`: Files to download (default: `Labels-v3.json Frames-v3.zip`)

### 2. `extract_frames.py`
Extracts frames from zip file and visualizes annotations.

**Basic Usage:**
```bash
# Extract and preview frames
python extract_frames.py "../../data/SoccerNet/england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea"

# Just extract
python extract_frames.py <match_dir> --extract

# Just preview
python extract_frames.py <match_dir> --preview --num_samples 10
```

**Arguments:**
- `match_dir`: Path to match directory (positional, required)
- `--extract`: Extract frames from zip
- `--preview`: Preview frames with annotations
- `--num_samples`: Number of frames to preview (default: 5)
- `--overwrite`: Overwrite existing extracted frames
- `--all`: Extract and preview (default if no flags specified)

## Complete Workflow

```bash
# Step 1: Download match data (requires specifying the split)
python download_match.py --split train

# Step 2: Extract and visualize
python extract_frames.py \
    "../../data/SoccerNet/england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea" \
    --num_samples 10
```

**Note:** You need to know which split (train/valid/test) contains your game. If unsure, check the [SoccerNet dataset documentation](https://github.com/SoccerNet/SoccerNet-v3).

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

## Requirements

Make sure these packages are installed:
```bash
pip install SoccerNet opencv-python tqdm numpy
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
