"""
NuScenes dataset loader for EVO-1 autonomous driving training

This module provides comprehensive data loading and preprocessing
for nuScenes dataset adapted for federated learning scenarios.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import cv2
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import logging
from dataclasses import dataclass

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import view_points
except ImportError:
    logging.warning("NuScenes SDK not found. Please install with: pip install nuscenes-devkit")

from ..utils.config import DataConfig, ModelConfig


@dataclass
class DrivingObservation:
    """Driving observation data structure"""
    # Multi-view images
    front_image: np.ndarray
    left_image: np.ndarray  
    right_image: np.ndarray
    
    # Vehicle state
    ego_pose: np.ndarray  # [x, y, z, roll, pitch, yaw]
    ego_velocity: np.ndarray  # [vx, vy, vz]
    ego_acceleration: np.ndarray  # [ax, ay, az]
    steering_angle: float
    throttle: float
    brake: float
    
    # Navigation context
    current_route: List[np.ndarray]  # waypoints
    target_waypoint: np.ndarray
    speed_limit: float
    
    # Environment context
    timestamp: float
    weather: str
    time_of_day: str
    
    # Language instruction
    instruction: str
    
    # Future trajectory (for training)
    future_trajectory: np.ndarray  # [T, 3] waypoints
    future_controls: np.ndarray  # [T, 3] [steering, throttle, brake]


class NuScenesDrivingLoader(Dataset):
    """NuScenes dataset loader for EVO-1 driving model"""
    
    def __init__(
        self,
        config: DataConfig,
        model_config: ModelConfig,
        split: str = "train",
        client_id: Optional[int] = None,
        num_clients: Optional[int] = None
    ):
        self.config = config
        self.model_config = model_config
        self.split = split
        
        # Initialize NuScenes
        try:
            self.nusc = NuScenes(
                version=config.version,
                dataroot=config.data_root,
                verbose=False
            )
        except Exception as e:
            logging.error(f"Failed to load NuScenes: {e}")
            raise
        
        # Load scenes and filter for driving scenarios
        self.scenes = self._filter_driving_scenes()
        
        # Federated learning: split data by client
        if client_id is not None and num_clients is not None:
            self.scenes = self._split_data_by_client(self.scenes, client_id, num_clients)
        
        # Preprocess sequences
        self.sequences = self._create_sequences()
        
        # Setup image transforms
        self.image_transform = self._setup_image_transforms()
        
        # Setup control normalization
        self._setup_normalization()
        
        logging.info(f"Loaded {len(self.sequences)} sequences for {split} split")
    
    def _filter_driving_scenes(self) -> List[Dict]:
        """Filter scenes for driving-relevant scenarios"""
        driving_scenes = []
        
        for scene in self.nusc.scene:
            # Filter by scene description for driving relevance
            if any(keyword in scene['description'].lower() 
                   for keyword in ['driving', 'vehicle', 'traffic', 'road', 'highway']):
                driving_scenes.append(scene)
        
        # Ensure we have enough driving scenes
        if len(driving_scenes) < 10:
            logging.warning(f"Only {len(driving_scenes)} driving scenes found. Using all scenes.")
            return self.nusc.scene[:100]  # Limit to first 100 scenes for memory
        
        return driving_scenes
    
    def _split_data_by_client(self, scenes: List[Dict], client_id: int, num_clients: int) -> List[Dict]:
        """Split scenes across federated learning clients"""
        scenes_per_client = len(scenes) // num_clients
        start_idx = client_id * scenes_per_client
        end_idx = start_idx + scenes_per_client if client_id < num_clients - 1 else len(scenes)
        
        return scenes[start_idx:end_idx]
    
    def _create_sequences(self) -> List[Tuple[str, int]]:
        """Create training sequences from scenes"""
        sequences = []
        
        for scene in self.scenes:
            scene_tokens = []
            
            # Get all samples in the scene
            sample = self.nusc.get('sample', scene['first_sample_token'])
            
            while sample['next'] != '':
                scene_tokens.append(sample['token'])
                sample = self.nusc.get('sample', sample['next'])
            scene_tokens.append(sample['token'])  # Last sample
            
            # Create sequences of specified length
            for i in range(0, len(scene_tokens) - self.config.sequence_length, self.config.sequence_stride):
                sequences.append((scene['name'], i))
        
        return sequences
    
    def _setup_image_transforms(self) -> transforms.Compose:
        """Setup image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((self.config.image_size[0], self.config.image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
    
    def _setup_normalization(self):
        """Setup control parameter normalization"""
        # Control ranges
        self.steering_range = (-self.config.max_steering, self.config.max_steering)
        self.throttle_range = (0.0, 1.0)
        self.brake_range = (0.0, 1.0)
        self.speed_range = (self.config.min_speed, self.config.max_speed)
        
        # Initialize instance variables for state tracking
        self._prev_ego_pose = None
        self._prev_velocity = None
        self._prev_timestamp = None
    
    def _normalize_control(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize control values to [-1, 1]"""
        return 2.0 * (value - min_val) / (max_val - min_val) - 1.0
    
    def _denormalize_control(self, value: float, min_val: float, max_val: float) -> float:
        """Denormalize control values from [-1, 1]"""
        return (value + 1.0) * (max_val - min_val) / 2.0 + min_val
    
    def _get_camera_image(self, sample_token: str, camera_channel: str) -> np.ndarray:
        """Get camera image for specified channel"""
        sample = self.nusc.get('sample', sample_token)
        sample_data = self.nusc.get('sample_data', sample['data'][camera_channel])
        
        # Load image
        image_path = os.path.join(self.nusc.dataroot, sample_data['filename'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _get_vehicle_state(self, sample_token: str) -> Dict[str, np.ndarray]:
        """Get vehicle state from sample"""
        sample = self.nusc.get('sample', sample_token)
        
        # Get ego pose from front camera sample data
        front_cam_token = sample['data']['CAM_FRONT']
        front_cam_data = self.nusc.get('sample_data', front_cam_token)
        ego_pose_token = front_cam_data['ego_pose_token']
        ego_pose_data = self.nusc.get('ego_pose', ego_pose_token)
        
        ego_pose = np.array([
            ego_pose_data['translation'][0],  # x
            ego_pose_data['translation'][1],  # y
            ego_pose_data['translation'][2],  # z
            ego_pose_data['rotation'][0],    # roll
            ego_pose_data['rotation'][1],    # pitch
            ego_pose_data['rotation'][2]     # yaw
        ])
        
        # Calculate velocity and acceleration (approximation)
        velocity = np.zeros(3)
        acceleration = np.zeros(3)
        
        if self._prev_ego_pose is not None:
            # Convert timestamp from microseconds to seconds
            dt = (ego_pose_data['timestamp'] - self._prev_timestamp) / 1e6
            if dt > 0:
                # Calculate velocity for all 3 axes
                velocity = (ego_pose[:3] - self._prev_ego_pose[:3]) / dt
                # Calculate acceleration for all 3 axes
                acceleration = (velocity - self._prev_velocity) / dt if self._prev_velocity is not None else np.zeros(3)
        
        # Update previous state
        self._prev_ego_pose = ego_pose.copy()
        self._prev_velocity = velocity.copy()
        self._prev_timestamp = ego_pose_data['timestamp']
        
        return {
            'pose': ego_pose,
            'velocity': velocity,
            'acceleration': acceleration
        }
    
    def _extract_controls_from_trajectory(self, current_pose: np.ndarray, 
                                         future_trajectory: np.ndarray) -> np.ndarray:
        """Extract steering, throttle, brake controls from trajectory"""
        controls = []
        
        for i in range(1, len(future_trajectory)):
            # Current and next positions
            curr_pos = future_trajectory[i-1][:2]
            next_pos = future_trajectory[i][:2]
            
            # Calculate desired heading
            desired_heading = np.arctan2(
                next_pos[1] - curr_pos[1],
                next_pos[0] - curr_pos[0]
            )
            
            # Current heading from pose
            current_heading = current_pose[5]  # yaw
            
            # Steering angle (normalized)
            steering_error = desired_heading - current_heading
            # Normalize to [-pi, pi]
            steering_error = np.arctan2(np.sin(steering_error), np.cos(steering_error))
            steering = self._normalize_control(
                steering_error, 
                -self.config.max_steering, 
                self.config.max_steering
            )
            
            # Distance and speed
            distance = np.linalg.norm(next_pos - curr_pos)
            desired_speed = min(distance * 10.0, self.config.max_speed)  # Approximation
            
            current_speed = np.linalg.norm(future_trajectory[i-1][3:6] if len(future_trajectory[i-1]) > 3 else np.zeros(3))
            
            # Throttle and brake
            if desired_speed > current_speed:
                throttle = self._normalize_control(
                    min((desired_speed - current_speed) / 5.0, 1.0),
                    0.0, 1.0
                )
                brake = -1.0  # No braking
            else:
                throttle = -1.0  # No throttle
                brake = self._normalize_control(
                    min((current_speed - desired_speed) / 5.0, 1.0),
                    0.0, 1.0
                )
            
            controls.append([steering, throttle, brake])
        
        return np.array(controls)
    
    def _generate_driving_instruction(self, scene_data: Dict, vehicle_state: Dict) -> str:
        """Generate natural language driving instruction"""
        current_speed = np.linalg.norm(vehicle_state['velocity'])
        speed_limit = np.random.uniform(20, 30)  # Mock speed limit
        
        instructions = [
            f"Drive forward maintaining {speed_limit:.1f} m/s speed",
            "Follow the lane ahead and stay in your lane",
            "Prepare to navigate the intersection safely",
            "Maintain safe following distance from vehicles ahead",
            "Adapt speed according to traffic conditions"
        ]
        
        # Context-aware instruction selection
        if current_speed < 5.0:
            return "Accelerate gently to reach cruising speed"
        elif current_speed > speed_limit * 1.2:
            return "Reduce speed to maintain speed limit"
        elif np.random.random() < 0.3:  # 30% chance for varied instruction
            return np.random.choice(instructions[1:])
        else:
            return instructions[0]
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get training sample"""
        scene_name, start_idx = self.sequences[idx]
        
        # Get scene
        scene = next(s for s in self.nusc.scene if s['name'] == scene_name)
        
        # Get sequence samples
        sequence_data = []
        current_sample_token = scene['first_sample_token']
        
        # Navigate to start position
        for _ in range(start_idx):
            sample = self.nusc.get('sample', current_sample_token)
            if sample['next'] != '':
                current_sample_token = sample['next']
        
        # Process sequence
        for i in range(self.config.sequence_length):
            sample = self.nusc.get('sample', current_sample_token)
            
            # Get multi-view images
            front_image = self._get_camera_image(current_sample_token, 'CAM_FRONT')
            left_image = self._get_camera_image(current_sample_token, 'CAM_FRONT_LEFT')
            right_image = self._get_camera_image(current_sample_token, 'CAM_FRONT_RIGHT')
            
            # Get vehicle state
            vehicle_state = self._get_vehicle_state(current_sample_token)
            
            # Generate future trajectory (mock for now)
            current_pos = vehicle_state['pose'][:2]
            heading = vehicle_state['pose'][5]
            
            # Simple straight-line trajectory with some variation
            future_trajectory = []
            for t in range(1, self.config.max_waypoints + 1):
                dt = t * 0.1  # 0.1 second intervals
                # Add small lateral variation for realism
                lateral_offset = np.sin(t * 0.5) * 0.5
                future_pos = current_pos + dt * 10.0 * np.array([np.cos(heading), np.sin(heading)])
                future_pos += lateral_offset * np.array([-np.sin(heading), np.cos(heading)])
                future_trajectory.append([future_pos[0], future_pos[1], 0.0])
            
            future_trajectory = np.array(future_trajectory)
            
            # Extract controls from trajectory
            controls = self._extract_controls_from_trajectory(vehicle_state['pose'], future_trajectory)
            
            # Generate instruction
            instruction = self._generate_driving_instruction(scene, vehicle_state)
            
            sequence_data.append({
                'front_image': front_image,
                'left_image': left_image,
                'right_image': right_image,
                'ego_pose': vehicle_state['pose'],
                'ego_velocity': vehicle_state['velocity'],
                'ego_acceleration': vehicle_state['acceleration'],
                'future_trajectory': future_trajectory,
                'future_controls': controls,
                'instruction': instruction,
                'timestamp': sample['timestamp']
            })
            
            # Move to next sample
            if sample['next'] != '':
                current_sample_token = sample['next']
        
        # Create training sample from sequence
        return self._create_training_sample(sequence_data)
    
    def _create_training_sample(self, sequence_data: List[Dict]) -> Dict[str, Any]:
        """Create training sample from sequence data"""
        # Use the first frame as current observation
        current_frame = sequence_data[0]
        
        # Stack images
        front_img = self.image_transform(Image.fromarray(current_frame['front_image']))
        left_img = self.image_transform(Image.fromarray(current_frame['left_image']))
        right_img = self.image_transform(Image.fromarray(current_frame['right_image']))
        
        images = torch.stack([front_img, left_img, right_img])  # [3, C, H, W]
        image_mask = torch.tensor([1.0, 1.0, 1.0])  # All cameras available
        
        # State vector (position, velocity, acceleration)
        state = torch.cat([
            torch.from_numpy(current_frame['ego_pose']).float(),
            torch.from_numpy(current_frame['ego_velocity']).float(),
            torch.from_numpy(current_frame['ego_acceleration']).float()
        ])  # [12] (6 pose + 3 vel + 3 acc)
        
        # Future controls (target for training)
        future_controls = torch.from_numpy(current_frame['future_controls']).float()
        
        # Trim to model's max_waypoints
        max_controls = min(self.model_config.max_waypoints, len(future_controls))
        future_controls = future_controls[:max_controls]
        
        # Pad if necessary
        if len(future_controls) < self.model_config.max_waypoints:
            padding = torch.zeros(self.model_config.max_waypoints - len(future_controls), 3)
            future_controls = torch.cat([future_controls, padding])
        
        return {
            'images': images,
            'image_mask': image_mask,
            'state': state,
            'future_controls': future_controls,
            'instruction': current_frame['instruction'],
            'timestamp': current_frame['timestamp']
        }


def create_dataloader(
    config: DataConfig,
    model_config: ModelConfig,
    split: str = "train",
    client_id: Optional[int] = None,
    num_clients: Optional[int] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create dataloader for NuScenes driving dataset"""
    
    dataset = NuScenesDrivingLoader(
        config=config,
        model_config=model_config,
        split=split,
        client_id=client_id,
        num_clients=num_clients
    )
    
    def collate_fn(batch):
        """Custom collate function for variable-length sequences"""
        # Stack tensors
        images = torch.stack([item['images'] for item in batch])  # [B, 3, C, H, W]
        image_mask = torch.stack([item['image_mask'] for item in batch])  # [B, 3]
        states = torch.stack([item['state'] for item in batch])  # [B, 12]
        future_controls = torch.stack([item['future_controls'] for item in batch])  # [B, T, 3]
        
        # Text instructions (keep as list for now)
        instructions = [item['instruction'] for item in batch]
        
        return {
            'images': images,
            'image_mask': image_mask,
            'state': states,
            'future_controls': future_controls,
            'instructions': instructions,
            'timestamps': [item['timestamp'] for item in batch]
        }
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=split == "train"
    )