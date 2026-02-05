#!/usr/bin/env python3
"""
Download SoccerNet-v3 frames and labels for a specific match.

This script downloads the annotated frames and labels needed for 3D reconstruction
from the SoccerNet-v3 dataset for a specified match.

Note: SoccerNet downloader downloads games by split. This script will download
all games in the split and verify the requested game exists.
"""

import argparse
from pathlib import Path
from typing import Optional
from SoccerNet.Downloader import SoccerNetDownloader


def find_game_path(base_dir: Path, game_name: str) -> Optional[Path]:
    """
    Search for a game directory by name in the SoccerNet directory structure.

    Args:
        base_dir: Base SoccerNet directory
        game_name: Game name to search for

    Returns:
        Path to game directory if found, None otherwise
    """
    # Search in all league/season subdirectories
    for game_path in base_dir.rglob(game_name):
        if game_path.is_dir():
            return game_path
    return None


def download_match(
    local_directory: str,
    league: str,
    season: str,
    game: str,
    files: Optional[list[str]] = None,
    split: Optional[str] = None,
):
    """
    Download SoccerNet data for a specific match.

    Args:
        local_directory: Path where to save the downloaded data
        league: League name (e.g., "england_epl")
        season: Season (e.g., "2016-2017")
        game: Game identifier in format "YYYY-MM-DD - HH-MM Team1 X - Y Team2"
        files: List of files to download (default: Frames-v3.zip and Labels-v3.json)
        split: Dataset split (train/valid/test). Required.
    """
    if files is None:
        files = ["Labels-v3.json", "Frames-v3.zip"]

    if split is None:
        raise ValueError(
            "Split must be specified (train/valid/test). "
            "Use --split to specify which split contains your game."
        )

    # Initialize downloader
    downloader = SoccerNetDownloader(LocalDirectory=local_directory)

    print(f"Downloading data for match: {game}")
    print(f"League: {league}, Season: {season}")
    print(f"Split: {split}")
    print(f"Files to download: {files}")
    print(f"Save location: {local_directory}")
    print("\nNote: SoccerNet downloads all games in the split.")
    print("Your requested game will be included if it exists in this split.\n")

    # Download games for the specified split
    try:
        downloader.downloadGames(
            files=files,
            split=[split],
            task="frames"
        )
        print(f"✓ Successfully downloaded from '{split}' split")
    except Exception as e:
        print(f"✗ Error downloading from '{split}' split: {e}")
        raise

    # Verify the specific game was downloaded
    game_path = Path(local_directory) / league / season / game

    if game_path.exists():
        print(f"\n✓ Game found at: {game_path}")

        # Check which files were downloaded
        downloaded_files = []
        for file in files:
            file_path = game_path / file
            if file_path.exists():
                size = file_path.stat().st_size / (1024 * 1024)  # MB
                downloaded_files.append(f"  • {file} ({size:.2f} MB)")

        if downloaded_files:
            print("\nDownloaded files:")
            print("\n".join(downloaded_files))

        return game_path
    else:
        print(f"\n✗ Warning: Game not found at expected path: {game_path}")
        print(f"\nSearching for game in downloaded data...")

        found_path = find_game_path(Path(local_directory), game)
        if found_path:
            print(f"✓ Found game at: {found_path}")
            return found_path
        else:
            print(f"✗ Game '{game}' not found in downloaded data.")
            print(f"   The game may not exist in the '{split}' split.")
            print(f"   Try a different split: train, valid, or test")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Download SoccerNet-v3 data for a specific match",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Leicester vs Chelsea from train split
  python download_match.py --split train

  # Download from a different league/season
  python download_match.py --league spain_laliga --season 2015-2016 \\
    --game "2016-04-02 - 17-00 Barcelona 1 - 2 Real Madrid" --split valid
        """,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/SoccerNet",
        help="Directory where to save the downloaded data (default: ./data/SoccerNet)",
    )
    parser.add_argument(
        "--league",
        type=str,
        default="england_epl",
        help="League name (default: england_epl)",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2016-2017",
        help="Season (default: 2016-2017)",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="2017-01-14 - 20-30 Leicester 0 - 3 Chelsea",
        help="Game identifier (default: 2017-01-14 - 20-30 Leicester 0 - 3 Chelsea)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test"],
        required=True,
        help="Dataset split (REQUIRED: train/valid/test)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["Labels-v3.json", "Frames-v3.zip"],
        help="Files to download (default: Labels-v3.json Frames-v3.zip)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Download the match
    game_path = download_match(
        local_directory=args.output_dir,
        league=args.league,
        season=args.season,
        game=args.game,
        files=args.files,
        split=args.split,
    )

    print("\n" + "=" * 60)
    if game_path:
        print("Download complete!")
        print(f"Data saved to: {game_path}")
    else:
        print("Download completed, but requested game not found.")
        print("Please verify the game name, league, season, and split.")
    print("=" * 60)


if __name__ == "__main__":
    main()
