"""
SoccerNet Data Parser for Nerfstudio.

Parses SoccerNet-v3D multi-view data into Nerfstudio format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class SoccerDataParserConfig(DataParserConfig):
    """SoccerNet data parser configuration."""

    _target: Type = field(default_factory=lambda: SoccerDataParser)

    data: Path = Path("data/SoccerNet")
    """Path to SoccerNet dataset root"""

    match_path: Optional[str] = None
    """Relative path to specific match (e.g., 'england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea')"""

    action_id: Optional[str] = None
    """Specific action to reconstruct (e.g., '0' for action 0). If None, uses first action."""

    use_v3d: bool = True
    """Use Labels-v3D.json with camera calibration"""

    scale_factor: float = 1.0
    """Scale factor for the scene"""

    orientation_method: Literal["pca", "up", "none"] = "up"
    """Method to orient the scene"""

    center_method: Literal["poses", "focus", "none"] = "poses"
    """Method to center the scene"""

    auto_scale_poses: bool = True
    """Automatically scale camera poses to fit in unit cube"""

    downscale_factor: int = 1
    """Image downscale factor (1 = full resolution)"""

    load_from_zip: bool = True
    """Load images directly from Frames-v3.zip"""


@dataclass
class SoccerDataParser(DataParser):
    """SoccerNet data parser for Nerfstudio.

    Converts SoccerNet-v3D format to Nerfstudio's camera and image format.
    """

    config: SoccerDataParserConfig

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        """Generate dataparser outputs for the given split.

        Args:
            split: Dataset split (train/val/test)

        Returns:
            DataparserOutputs containing cameras and metadata
        """
        # Determine match directory
        if self.config.match_path:
            match_dir = self.config.data / self.config.match_path
        else:
            # Find first available match
            match_dir = self._find_first_match()

        if not match_dir.exists():
            raise ValueError(f"Match directory not found: {match_dir}")

        # Load labels
        labels_file = "Labels-v3D.json" if self.config.use_v3d else "Labels-v3.json"
        labels_path = match_dir / labels_file

        if not labels_path.exists():
            # Fallback to v3 if v3D not available
            if self.config.use_v3d:
                labels_path = match_dir / "Labels-v3.json"
                print(f"Warning: Labels-v3D.json not found, using Labels-v3.json")

        if not labels_path.exists():
            raise ValueError(f"Labels file not found: {labels_path}")

        labels = load_from_json(labels_path)

        # Get action frames
        action_id = self.config.action_id or labels["GameMetadata"]["list_actions"][0]

        if action_id not in labels["actions"]:
            raise ValueError(f"Action {action_id} not found in labels")

        action_data = labels["actions"][action_id]

        # Collect all frames (action + replays)
        frame_names = [action_id] + action_data.get("linked_replays", [])

        # Parse cameras and images
        cameras_list = []
        image_filenames = []

        for frame_name in frame_names:
            # Determine if action or replay
            if frame_name == action_id:
                frame_data = labels["actions"][frame_name]
            else:
                frame_data = labels["replays"][frame_name]

            # Get image metadata
            img_meta = frame_data["imageMetadata"]
            width, height = img_meta["width"], img_meta["height"]

            # Apply downscaling
            if self.config.downscale_factor > 1:
                width = width // self.config.downscale_factor
                height = height // self.config.downscale_factor

            # Get camera calibration (if available)
            if "calibration" in frame_data and self.config.use_v3d:
                camera = self._parse_v3d_calibration(frame_data["calibration"], width, height)
            else:
                # Use default camera parameters (will need optimization)
                camera = self._create_default_camera(width, height)

            cameras_list.append(camera)

            # Store image path
            image_path = match_dir / "frames" / f"{frame_name}.png"
            if not image_path.exists():
                # Try in zip file
                image_path = match_dir / "Frames-v3.zip"

            image_filenames.append(image_path)

        # Stack cameras
        cameras = self._stack_cameras(cameras_list)

        # Create scene box
        scene_box = self._create_scene_box()

        # Split into train/val
        num_images = len(image_filenames)
        indices = np.arange(num_images)

        if split == "train":
            # Use all but one for training
            indices = indices[:-1] if num_images > 1 else indices
        else:  # val
            # Use last image for validation
            indices = indices[-1:] if num_images > 1 else indices

        train_indices = indices if split == "train" else np.array([])
        val_indices = indices if split != "train" else np.array([])

        return DataparserOutputs(
            image_filenames=[image_filenames[i] for i in indices],
            cameras=cameras[indices],
            scene_box=scene_box,
            dataparser_scale=self.config.scale_factor,
            metadata={
                "match_path": str(match_dir),
                "action_id": action_id,
                "num_views": len(frame_names),
                "labels": labels,
            },
        )

    def _find_first_match(self) -> Path:
        """Find the first available match in the dataset."""
        # Search for Labels-v3.json or Labels-v3D.json files
        labels_files = list(self.config.data.rglob("Labels-v3*.json"))

        if not labels_files:
            raise ValueError(f"No SoccerNet matches found in {self.config.data}")

        return labels_files[0].parent

    def _parse_v3d_calibration(
        self, calibration: dict, width: int, height: int
    ) -> Cameras:
        """Parse SoccerNet-v3D calibration to Nerfstudio camera.

        Args:
            calibration: Calibration dictionary from Labels-v3D.json
            width: Image width
            height: Image height

        Returns:
            Cameras object
        """
        # Extract camera parameters
        fx = calibration.get("x_focal_length", width)
        fy = calibration.get("y_focal_length", width)
        cx = calibration.get("principal_point", [width / 2, height / 2])[0]
        cy = calibration.get("principal_point", [width / 2, height / 2])[1]

        # Camera position and rotation
        position = calibration.get("position", [0, 0, 5])
        rotation_matrix = calibration.get("rotation_matrix", np.eye(3).tolist())

        # Convert to camera-to-world transform
        c2w = np.eye(4)
        c2w[:3, :3] = rotation_matrix
        c2w[:3, 3] = position

        # Convert to torch
        c2w = torch.from_numpy(c2w).float()
        fx = torch.tensor([fx], dtype=torch.float32)
        fy = torch.tensor([fy], dtype=torch.float32)
        cx = torch.tensor([cx], dtype=torch.float32)
        cy = torch.tensor([cy], dtype=torch.float32)

        return Cameras(
            camera_to_worlds=c2w[:3, :4].unsqueeze(0),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=torch.tensor([width]),
            height=torch.tensor([height]),
            camera_type=CameraType.PERSPECTIVE,
        )

    def _create_default_camera(self, width: int, height: int) -> Cameras:
        """Create a default camera when calibration is unavailable.

        Args:
            width: Image width
            height: Image height

        Returns:
            Cameras object with estimated parameters
        """
        # Default focal length (assuming ~50mm equivalent)
        fx = fy = width * 1.2
        cx, cy = width / 2, height / 2

        # Default pose (looking at origin from above)
        c2w = torch.eye(4)
        c2w[2, 3] = 10.0  # 10 meters above field

        return Cameras(
            camera_to_worlds=c2w[:3, :4].unsqueeze(0),
            fx=torch.tensor([fx], dtype=torch.float32),
            fy=torch.tensor([fy], dtype=torch.float32),
            cx=torch.tensor([cx], dtype=torch.float32),
            cy=torch.tensor([cy], dtype=torch.float32),
            width=torch.tensor([width]),
            height=torch.tensor([height]),
            camera_type=CameraType.PERSPECTIVE,
        )

    def _stack_cameras(self, cameras_list: list) -> Cameras:
        """Stack individual cameras into batched Cameras object.

        Args:
            cameras_list: List of individual Cameras

        Returns:
            Batched Cameras object
        """
        # Extract all camera parameters
        c2ws = torch.cat([c.camera_to_worlds for c in cameras_list], dim=0)
        fxs = torch.cat([c.fx for c in cameras_list], dim=0)
        fys = torch.cat([c.fy for c in cameras_list], dim=0)
        cxs = torch.cat([c.cx for c in cameras_list], dim=0)
        cys = torch.cat([c.cy for c in cameras_list], dim=0)
        widths = torch.cat([c.width for c in cameras_list], dim=0)
        heights = torch.cat([c.height for c in cameras_list], dim=0)

        return Cameras(
            camera_to_worlds=c2ws,
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            width=widths,
            height=heights,
            camera_type=CameraType.PERSPECTIVE,
        )

    def _create_scene_box(self) -> SceneBox:
        """Create scene box for soccer field.

        Returns:
            SceneBox centered on field
        """
        # Standard soccer field dimensions
        field_length = 105.0
        field_width = 68.0
        height = 10.0

        aabb = torch.tensor(
            [
                [-field_length / 2, -field_width / 2, 0.0],
                [field_length / 2, field_width / 2, height],
            ],
            dtype=torch.float32,
        )

        return SceneBox(aabb=aabb)
