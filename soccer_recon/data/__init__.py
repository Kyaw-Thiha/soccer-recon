"""
SoccerNet data management utilities.

This module provides tools for downloading and processing SoccerNet-v3 dataset.
"""

from pathlib import Path

__version__ = "0.1.0"

# Default data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "SoccerNet"
