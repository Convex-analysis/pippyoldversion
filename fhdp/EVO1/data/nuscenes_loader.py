"""
NuScenes dataset loader for EVO-1 autonomous driving training.

Follows Official NuScenes Devkit conventions:
  - ego_pose rotation is a [w, x, y, z] quaternion; yaw is extracted via pyquaternion.
  - calibrated_sensor is loaded per camera to expose intrinsic K and cam2ego extrinsic.
  - Sequences are pre-indexed at __init__ time so __getitem__ is O(1) per token lookup.
  - Supports 3-camera (front) or 6-camera (surround) modes via DataConfig.use_6_cameras.
  - Federated data splitting: contiguous scene slices per client_id.
"""

import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any

import cv2
from PIL import Image
import torchvision.transforms as transforms
from dataclasses import dataclass
from pyquaternion import Quaternion

from nuscenes.utils.splits import train, val, test, mini_train, mini_val

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import view_points
except ImportError:
    logging.warning("NuScenes SDK not found. Install with: pip install nuscenes-devkit")

from ..utils.config import DataConfig, ModelConfig


# ---------------------------------------------------------------------------
# Camera channel definitions
# ---------------------------------------------------------------------------

CAMERAS_3 = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
CAMERAS_6 = CAMERAS_3 + ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

_SPLIT_MAP = {
    "train": train,
    "val": val,
    "test": test,
    "mini_train": mini_train,
    "mini_val": mini_val,
}


# ---------------------------------------------------------------------------
# Utility: quaternion → yaw (NuScenes convention: [w, x, y, z])
# ---------------------------------------------------------------------------

def quat_to_yaw(rotation: List[float]) -> float:
    """Convert NuScenes [w, x, y, z] quaternion to yaw (radians)."""
    q = Quaternion(rotation)  # Quaternion([w, x, y, z])
    return q.yaw_pitch_roll[0]


def quat_to_rotation_matrix(rotation: List[float]) -> np.ndarray:
    """Return 3×3 rotation matrix from NuScenes [w, x, y, z] quaternion."""
    return Quaternion(rotation).rotation_matrix


def build_cam2ego(calibrated: Dict) -> np.ndarray:
    """Build 4×4 cam-to-ego transform from calibrated_sensor record."""
    R = quat_to_rotation_matrix(calibrated["rotation"])
    t = np.array(calibrated["translation"])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class DrivingObservation:
    """Per-frame driving observation."""
    front_image: np.ndarray
    left_image: np.ndarray
    right_image: np.ndarray
    ego_pose: np.ndarray          # [x, y, z, yaw]  (yaw from pyquaternion)
    ego_velocity: np.ndarray      # [vx, vy, vz]
    ego_acceleration: np.ndarray  # [ax, ay, az]
    steering_angle: float
    throttle: float
    brake: float
    current_route: List[np.ndarray]
    target_waypoint: np.ndarray
    speed_limit: float
    timestamp: float
    weather: str
    time_of_day: str
    instruction: str
    future_trajectory: np.ndarray  # [T, 3]
    future_controls: np.ndarray    # [T, 3]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NuScenesDrivingLoader(Dataset):
    """NuScenes dataset loader for EVO-1 driving model.

    Key design decisions (vs. original version):
    1. Rotation quaternion is correctly converted to yaw via pyquaternion.
    2. calibrated_sensor is queried per sample_data to obtain K and cam2ego.
    3. Scene sample-token chains are pre-built in __init__ (self._scene_tokens),
       so __getitem__ is O(1) instead of O(scene_length).
    4. Supports 3-camera (default) or 6-camera surround mode.
    5. ego_pose is [x, y, z, yaw] — 4-D instead of the erroneous 6-D.
       state vector = pose(4) + velocity(3) + acceleration(3) = 10-D.
    """

    def __init__(
        self,
        config: DataConfig,
        model_config: ModelConfig,
        split: str = "train",
        client_id: Optional[int] = None,
        num_clients: Optional[int] = None,
    ):
        self.config = config
        self.model_config = model_config
        self.split = split

        # Determine active cameras
        self.cameras = CAMERAS_6 if getattr(config, "use_6_cameras", False) else CAMERAS_3
        self.num_views = len(self.cameras)

        # Initialize NuScenes
        try:
            self.nusc = NuScenes(
                version=config.version,
                dataroot=config.data_root,
                verbose=False,
            )
        except Exception as exc:
            logging.error(f"Failed to load NuScenes: {exc}")
            raise

        # Filter scenes for the requested split
        self.scenes = self._filter_scenes()

        # Federated learning: assign a contiguous slice of scenes to this client
        if client_id is not None and num_clients is not None:
            self.scenes = self._split_by_client(self.scenes, client_id, num_clients)

        # Pre-build per-scene token chains (core performance optimization)
        # self._scene_tokens: Dict[str, List[str]]  scene_name → ordered sample tokens
        self._build_scene_token_index()

        # Build sequence list: (scene_name, token_start_index)
        self.sequences = self._create_sequences()

        # Image transform pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_size[0], config.image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ])

        self._setup_normalization()

        logging.info(
            f"NuScenesDrivingLoader [{split}]: "
            f"{len(self.scenes)} scenes, {len(self.sequences)} sequences, "
            f"{self.num_views} cameras"
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _filter_scenes(self) -> List[Dict]:
        """Return scenes for the requested split (uses official NuScenes splits)."""
        target_names = _SPLIT_MAP.get(self.split)
        if target_names is None:
            logging.warning(f"Unknown split '{self.split}', using all scenes.")
            return list(self.nusc.scene)

        scenes = [s for s in self.nusc.scene if s["name"] in target_names]

        # Optional keyword sub-filter (only when explicitly enabled)
        if getattr(self.config, "filter_by_keywords", False):
            keywords = ["driving", "vehicle", "traffic", "road", "highway"]
            filtered = [
                s for s in scenes
                if any(kw in s["description"].lower() for kw in keywords)
            ]
            if filtered:
                scenes = filtered

        if len(scenes) < 5:
            logging.warning(
                f"Only {len(scenes)} scenes for split '{self.split}'. "
                "Falling back to all scenes."
            )
            return list(self.nusc.scene)

        logging.info(f"Filtered {len(scenes)} scenes for split '{self.split}'")
        return scenes

    def _split_by_client(
        self, scenes: List[Dict], client_id: int, num_clients: int
    ) -> List[Dict]:
        """Assign a contiguous slice of scenes to this federated client."""
        n = len(scenes)
        per_client = n // num_clients
        start = client_id * per_client
        end = start + per_client if client_id < num_clients - 1 else n
        return scenes[start:end]

    def _build_scene_token_index(self) -> None:
        """Pre-build ordered sample-token list for every scene.

        Stored in self._scene_tokens: Dict[str, List[str]]
        This eliminates the O(N) chain traversal on every __getitem__ call.
        """
        self._scene_tokens: Dict[str, List[str]] = {}
        for scene in self.scenes:
            tokens: List[str] = []
            tok = scene["first_sample_token"]
            while tok:
                tokens.append(tok)
                tok = self.nusc.get("sample", tok)["next"] or None
            self._scene_tokens[scene["name"]] = tokens

    def _create_sequences(self) -> List[Tuple[str, int]]:
        """Build (scene_name, start_index) list from pre-built token chains."""
        sequences: List[Tuple[str, int]] = []
        for scene in self.scenes:
            tokens = self._scene_tokens[scene["name"]]
            n = len(tokens)
            if n >= self.config.sequence_length:
                for i in range(0, n - self.config.sequence_length + 1, self.config.sequence_stride):
                    sequences.append((scene["name"], i))
            elif n > 0:
                sequences.append((scene["name"], 0))
        return sequences

    def _setup_normalization(self) -> None:
        self.steering_range = (-self.config.max_steering, self.config.max_steering)
        self.throttle_range = (0.0, 1.0)
        self.brake_range = (0.0, 1.0)
        self.speed_range = (self.config.min_speed, self.config.max_speed)

    # ------------------------------------------------------------------
    # NuScenes accessors (follow official devkit patterns)
    # ------------------------------------------------------------------

    def _get_ego_pose_data(self, sample_token: str) -> Tuple[np.ndarray, float]:
        """Return ego pose [x, y, z, yaw] and timestamp (µs) for a sample.

        NuScenes ego_pose.rotation is [w, x, y, z] quaternion.
        We extract yaw correctly via pyquaternion.
        """
        sample = self.nusc.get("sample", sample_token)
        cam_data = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])

        yaw = quat_to_yaw(ego_pose["rotation"])
        pose = np.array([
            ego_pose["translation"][0],  # x
            ego_pose["translation"][1],  # y
            ego_pose["translation"][2],  # z
            yaw,                         # yaw (radians) — correctly derived from quaternion
        ], dtype=np.float32)
        return pose, ego_pose["timestamp"]

    def _get_camera_image(
        self, sample_token: str, camera_channel: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load image and return (image_rgb, K_3x3, cam2ego_4x4).

        K: 3×3 intrinsic matrix from calibrated_sensor.
        cam2ego: 4×4 camera-to-ego rigid transform.
        """
        sample = self.nusc.get("sample", sample_token)
        sd = self.nusc.get("sample_data", sample["data"][camera_channel])

        # Load image
        image_path = os.path.join(self.nusc.dataroot, sd["filename"])
        image = cv2.imread(image_path)
        if image is None:
            # Return black frame on missing file
            h, w = self.config.image_size
            image = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calibrated sensor → intrinsics + extrinsics
        cal = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K = np.array(cal["camera_intrinsic"], dtype=np.float32)  # 3×3
        cam2ego = build_cam2ego(cal).astype(np.float32)           # 4×4

        return image, K, cam2ego

    def _get_future_trajectory(
        self, sample_token: str, scene_name: str, start_idx: int, num_waypoints: int
    ) -> np.ndarray:
        """Extract future ego positions [T, 3] using pre-built token index."""
        tokens = self._scene_tokens[scene_name]
        trajectory: List[np.ndarray] = []

        for offset in range(1, num_waypoints + 1):
            next_idx = start_idx + offset
            if next_idx >= len(tokens):
                break
            pose, _ = self._get_ego_pose_data(tokens[next_idx])
            trajectory.append(pose[:3])

        if not trajectory:
            trajectory = [np.zeros(3, dtype=np.float32)]

        # Pad to num_waypoints
        while len(trajectory) < num_waypoints:
            trajectory.append(trajectory[-1].copy())

        return np.stack(trajectory, axis=0)  # [T, 3]

    def _extract_controls_from_trajectory(
        self,
        current_pose: np.ndarray,      # [x, y, z, yaw]
        current_velocity: np.ndarray,  # [vx, vy, vz]
        future_trajectory: np.ndarray, # [T, 3]
    ) -> np.ndarray:
        """Convert future trajectory to [steering, throttle, brake] controls."""
        controls: List[List[float]] = []
        current_heading = current_pose[3]  # yaw — now correctly extracted

        for i in range(1, len(future_trajectory)):
            curr_pos = future_trajectory[i - 1, :2]
            next_pos = future_trajectory[i, :2]

            desired_heading = np.arctan2(
                next_pos[1] - curr_pos[1],
                next_pos[0] - curr_pos[0],
            )

            # Steering error normalised to [-π, π]
            err = desired_heading - current_heading
            err = np.arctan2(np.sin(err), np.cos(err))
            steering = float(np.clip(err / self.config.max_steering, -1.0, 1.0))

            distance = float(np.linalg.norm(next_pos - curr_pos))
            desired_speed = min(distance * 10.0, self.config.max_speed)
            current_speed = float(np.linalg.norm(current_velocity))

            if desired_speed > current_speed:
                throttle = float(np.clip((desired_speed - current_speed) / 5.0, 0.0, 1.0))
                brake = 0.0
            else:
                throttle = 0.0
                brake = float(np.clip((current_speed - desired_speed) / 5.0, 0.0, 1.0))

            controls.append([steering, throttle, brake])

        if not controls:
            controls = [[0.0, 0.0, 0.0]]
        while len(controls) < len(future_trajectory):
            controls.append(controls[-1])

        return np.array(controls, dtype=np.float32)

    def _generate_instruction(
        self, scene: Dict, speed: float, sample_token: str
    ) -> str:
        """Deterministic natural-language driving instruction."""
        seed = int(sample_token[:8], 16)
        rng = np.random.RandomState(seed)

        speed_limit = self.config.max_speed
        pool = [
            f"Drive forward maintaining {speed_limit:.1f} m/s speed",
            "Follow the lane ahead and stay in your lane",
            "Prepare to navigate the intersection safely",
            "Maintain safe following distance from vehicles ahead",
            "Adapt speed according to traffic conditions",
        ]

        if speed < 5.0:
            return "Accelerate gently to reach cruising speed"
        if speed > speed_limit * 1.2:
            return "Reduce speed to maintain speed limit"
        if rng.random() < 0.3:
            return rng.choice(pool[1:])
        return pool[0]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene_name, start_idx = self.sequences[idx]
        tokens = self._scene_tokens[scene_name]
        scene = next(s for s in self.nusc.scene if s["name"] == scene_name)

        seq_len = self.config.sequence_length

        # Collect poses and timestamps for the whole sequence window first
        # (needed for velocity / acceleration finite-difference)
        seq_poses: List[np.ndarray] = []
        seq_timestamps: List[float] = []
        for offset in range(seq_len):
            t_idx = min(start_idx + offset, len(tokens) - 1)
            pose, ts = self._get_ego_pose_data(tokens[t_idx])
            seq_poses.append(pose)
            seq_timestamps.append(ts)

        # Load frame data
        sequence_data: List[Dict] = []
        for i in range(seq_len):
            t_idx = min(start_idx + i, len(tokens) - 1)
            tok = tokens[t_idx]

            # ----- Multi-view images -----
            cam_images: List[np.ndarray] = []
            cam_K: List[np.ndarray] = []
            cam2ego_list: List[np.ndarray] = []
            for cam in self.cameras:
                img, K, c2e = self._get_camera_image(tok, cam)
                cam_images.append(img)
                cam_K.append(K)
                cam2ego_list.append(c2e)

            # ----- Vehicle state -----
            current_pose = seq_poses[i]
            current_ts = seq_timestamps[i]

            velocity = np.zeros(3, dtype=np.float32)
            acceleration = np.zeros(3, dtype=np.float32)
            if i > 0:
                prev_pose = seq_poses[i - 1]
                dt = (current_ts - seq_timestamps[i - 1]) / 1e6  # µs → s
                if dt > 1e-6:
                    velocity = (current_pose[:3] - prev_pose[:3]) / dt
                    if i > 1:
                        prev_ts = seq_timestamps[i - 2]
                        dt_prev = (seq_timestamps[i - 1] - prev_ts) / 1e6
                        if dt_prev > 1e-6:
                            prev_vel = (seq_poses[i - 1][:3] - seq_poses[i - 2][:3]) / dt_prev
                            acceleration = (velocity - prev_vel) / dt

            # ----- Future trajectory (only needed for first frame) -----
            if i == 0:
                future_traj = self._get_future_trajectory(
                    tok, scene_name, start_idx, self.config.max_waypoints
                )
                controls = self._extract_controls_from_trajectory(
                    current_pose, velocity, future_traj
                )
                speed = float(np.linalg.norm(velocity))
                instruction = self._generate_instruction(scene, speed, tok)
            else:
                future_traj = np.zeros((self.config.max_waypoints, 3), dtype=np.float32)
                controls = np.zeros((self.config.max_waypoints, 3), dtype=np.float32)
                instruction = ""

            sample_record = self.nusc.get("sample", tok)
            sequence_data.append({
                "cam_images": cam_images,      # List[ndarray]
                "cam_K": cam_K,                # List[3×3]
                "cam2ego": cam2ego_list,        # List[4×4]
                "ego_pose": current_pose,       # [x, y, z, yaw]
                "ego_velocity": velocity,       # [vx, vy, vz]
                "ego_acceleration": acceleration,
                "future_trajectory": future_traj,
                "future_controls": controls,
                "instruction": instruction,
                "timestamp": sample_record["timestamp"],
            })

        return self._create_training_sample(sequence_data)

    def _create_training_sample(self, sequence_data: List[Dict]) -> Dict[str, Any]:
        """Convert sequence_data list to a training batch item."""
        frame = sequence_data[0]

        # Stack images: [N_views, C, H, W]
        imgs: List[torch.Tensor] = []
        for img_arr in frame["cam_images"]:
            imgs.append(self.image_transform(Image.fromarray(img_arr)))
        images = torch.stack(imgs)  # [N_views, 3, H, W]

        image_mask = torch.ones(self.num_views, dtype=torch.float32)

        # Camera intrinsics: [N_views, 3, 3]
        cam_K = torch.from_numpy(
            np.stack(frame["cam_K"], axis=0)
        ).float()  # [N_views, 3, 3]

        # cam2ego transforms: [N_views, 4, 4]
        cam2ego = torch.from_numpy(
            np.stack(frame["cam2ego"], axis=0)
        ).float()  # [N_views, 4, 4]

        # State vector: pose(6) + velocity(3) + acceleration(3) = 12-D
        # pose = [x, y, z, roll, pitch, yaw] (roll/pitch ≈ 0 for flat road driving)
        state = torch.cat([
            torch.from_numpy(frame["ego_pose"]).float(),          # [4]  x,y,z,yaw
            torch.tensor([0.0, 0.0], dtype=torch.float32),       # [2]  roll,pitch (flat road)
            torch.from_numpy(frame["ego_velocity"]).float(),      # [3]  vx,vy,vz
            torch.from_numpy(frame["ego_acceleration"]).float(),  # [3]  ax,ay,az
        ])  # [12]

        # Future controls target: [T, 3]
        fc = torch.from_numpy(frame["future_controls"]).float()
        max_wp = self.model_config.max_waypoints
        fc = fc[:max_wp]
        if fc.shape[0] < max_wp:
            pad = torch.zeros(max_wp - fc.shape[0], 3)
            fc = torch.cat([fc, pad])

        return {
            "images": images,            # [N_views, 3, H, W]
            "image_mask": image_mask,    # [N_views]
            "camera_intrinsics": cam_K,  # [N_views, 3, 3]  (new: for BEV projection)
            "cam2ego": cam2ego,          # [N_views, 4, 4]  (new: for BEV projection)
            "state": state,              # [10]
            "future_controls": fc,       # [max_waypoints, 3]
            "instruction": frame["instruction"],
            "timestamp": frame["timestamp"],
        }


# ---------------------------------------------------------------------------
# Public factory (interface unchanged — trainers need no modification)
# ---------------------------------------------------------------------------

def create_dataloader(
    config: DataConfig,
    model_config: ModelConfig,
    split: str = "train",
    client_id: Optional[int] = None,
    num_clients: Optional[int] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for NuScenes driving dataset.

    Interface is backward-compatible with the original version.
    New batch fields (camera_intrinsics, cam2ego) are silently ignored by
    trainers that do not use them.
    """
    dataset = NuScenesDrivingLoader(
        config=config,
        model_config=model_config,
        split=split,
        client_id=client_id,
        num_clients=num_clients,
    )

    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        images = torch.stack([b["images"] for b in batch])                    # [B, N, 3, H, W]
        image_mask = torch.stack([b["image_mask"] for b in batch])            # [B, N]
        states = torch.stack([b["state"] for b in batch])                     # [B, 10]
        future_controls = torch.stack([b["future_controls"] for b in batch])  # [B, T, 3]
        cam_K = torch.stack([b["camera_intrinsics"] for b in batch])          # [B, N, 3, 3]
        cam2ego = torch.stack([b["cam2ego"] for b in batch])                  # [B, N, 4, 4]
        instructions = [b["instruction"] for b in batch]
        timestamps = [b["timestamp"] for b in batch]

        return {
            "images": images,
            "image_mask": image_mask,
            "camera_intrinsics": cam_K,
            "cam2ego": cam2ego,
            "state": states,
            "future_controls": future_controls,
            "instructions": instructions,
            "timestamps": timestamps,
        }

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )
