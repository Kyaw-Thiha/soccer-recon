#!/usr/bin/env python3
"""
Check which matches have both v3D annotations and v3 frames.

This helps identify which matches are ready for training with camera calibration.
"""

import argparse
from pathlib import Path
from collections import defaultdict


def check_data_availability(
    v3d_dir: Path = Path("data/SoccerNet-v3D"),
    v3_dir: Path = Path("data/SoccerNet"),
):
    """Check which matches have both v3D labels and v3 frames.

    Args:
        v3d_dir: Path to SoccerNet-v3D directory (has calibration)
        v3_dir: Path to SoccerNet directory (has frames)
    """
    print("=" * 80)
    print("SoccerNet Data Availability Check")
    print("=" * 80)

    # Find all v3D matches (with calibration)
    v3d_labels = list(v3d_dir.rglob("Labels-v3D.json"))
    v3d_matches = {label.parent.relative_to(v3d_dir) for label in v3d_labels}

    # Find all v3 matches (with frames)
    v3_frames = list(v3_dir.rglob("Frames-v3.zip"))
    v3_matches = {frame.parent.relative_to(v3_dir) for frame in v3_frames}

    # Find matches with both
    complete_matches = v3d_matches & v3_matches
    v3d_only = v3d_matches - v3_matches
    v3_only = v3_matches - v3d_matches

    print(f"\n📊 Summary:")
    print(f"  v3D labels (with calibration):  {len(v3d_matches)} matches")
    print(f"  v3 frames:                       {len(v3_matches)} matches")
    print(f"  ✓ Complete (both):               {len(complete_matches)} matches")
    print(f"  ⚠ v3D only (no frames):          {len(v3d_only)} matches")
    print(f"  ⚠ v3 only (no calibration):      {len(v3_only)} matches")

    # Organize by league
    leagues = defaultdict(list)
    for match in sorted(complete_matches):
        league = match.parts[0] if match.parts else "unknown"
        leagues[league].append(match)

    print(f"\n✓ Ready for Training ({len(complete_matches)} matches):")
    print("-" * 80)
    for league, matches in sorted(leagues.items()):
        print(f"\n{league} ({len(matches)} matches):")
        for match in matches[:5]:  # Show first 5
            print(f"  • {match}")
        if len(matches) > 5:
            print(f"  ... and {len(matches) - 5} more")

    # Show some examples of incomplete matches
    if v3d_only:
        print(f"\n⚠ Examples with v3D labels but NO frames:")
        print("-" * 80)
        for match in sorted(v3d_only)[:5]:
            print(f"  • {match}")
        if len(v3d_only) > 5:
            print(f"  ... and {len(v3d_only) - 5} more")
        print(f"\n  → These have calibration but need frames from v3 dataset")

    if v3_only:
        print(f"\n⚠ Examples with v3 frames but NO calibration:")
        print("-" * 80)
        for match in sorted(v3_only)[:5]:
            print(f"  • {match}")
        if len(v3_only) > 5:
            print(f"  ... and {len(v3_only) - 5} more")
        print(f"\n  → These can train but without camera calibration")

    # Export complete matches list
    output_file = Path("data/ready_matches.txt")
    with open(output_file, "w") as f:
        for match in sorted(complete_matches):
            f.write(f"{match}\n")

    print(f"\n✓ Complete matches list saved to: {output_file}")
    print("=" * 80)

    return complete_matches, v3d_only, v3_only


def main():
    parser = argparse.ArgumentParser(
        description="Check SoccerNet data availability"
    )
    parser.add_argument(
        "--v3d-dir",
        type=Path,
        default=Path("data/SoccerNet-v3D"),
        help="Path to SoccerNet-v3D directory",
    )
    parser.add_argument(
        "--v3-dir",
        type=Path,
        default=Path("data/SoccerNet"),
        help="Path to SoccerNet directory",
    )

    args = parser.parse_args()

    check_data_availability(args.v3d_dir, args.v3_dir)


if __name__ == "__main__":
    main()
