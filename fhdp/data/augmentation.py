"""
Autonomous driving specific data augmentation strategies

This module provides comprehensive augmentation techniques for driving scenarios,
including weather simulation, traffic variations, and sensor noise modeling.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random
from typing import Dict, Tuple, List, Any
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter
import logging


class DrivingAugmentation:
    """Comprehensive augmentation suite for autonomous driving data"""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (448, 448),
        augment_probability: float = 0.8,
        weather_augmentation: bool = True,
        traffic_augmentation: bool = True,
        sensor_augmentation: bool = True
    ):
        self.image_size = image_size
        self.augment_probability = augment_probability
        self.weather_augmentation = weather_augmentation
        self.traffic_augmentation = traffic_augmentation
        self.sensor_augmentation = sensor_augmentation
        
        # Setup augmentations
        self._setup_weather_augmentations()
        self._setup_geometric_augmentations()
        self._setup_sensor_augmentations()
    
    def _setup_weather_augmentations(self):
        """Setup weather and lighting augmentations"""
        self.weather_transforms = {
            'rain': RainAugmentation(),
            'fog': FogAugmentation(), 
            'snow': SnowAugmentation(),
            'night': NightAugmentation(),
            'glare': GlareAugmentation(),
            'shadow': ShadowAugmentation()
        }
        
        self.lighting_transforms = T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        )
    
    def _setup_geometric_augmentations(self):
        """Setup geometric transformations"""
        self.geometric_transforms = T.Compose([
            T.RandomResizedCrop(
                self.image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=2
            )
        ])
    
    def _setup_sensor_augmentations(self):
        """Setup sensor noise and artifact augmentations"""
        self.sensor_transforms = {
            'gaussian_noise': GaussianNoise(),
            'motion_blur': MotionBlur(),
            'chromatic_aberration': ChromaticAberration(),
            'compression': CompressionArtifacts(),
            'lens_distortion': LensDistortion()
        }
    
    def augment_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Augment a batch of training data"""
        if random.random() > self.augment_probability:
            return batch
        
        augmented_batch = batch.copy()
        
        # Augment images
        if 'images' in batch:
            augmented_images = []
            for i in range(batch['images'].shape[0]):  # Batch dimension
                sample_images = batch['images'][i]  # [3, C, H, W]
                augmented_sample = []
                
                for j in range(sample_images.shape[0]):  # Camera views
                    img = sample_images[j]
                    img_aug = self._augment_single_image(img)
                    augmented_sample.append(img_aug)
                
                augmented_images.append(torch.stack(augmented_sample))
            
            augmented_batch['images'] = torch.stack(augmented_images)
        
        # Augment controls
        if 'future_controls' in batch:
            augmented_batch['future_controls'] = self._augment_controls(batch['future_controls'])
        
        # Augment state
        if 'state' in batch:
            augmented_batch['state'] = self._augment_state(batch['state'])
        
        return augmented_batch
    
    def _augment_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """Augment a single image tensor"""
        # Save original device
        original_device = image.device
        
        # Convert to PIL for some augmentations
        img_pil = T.ToPILImage()(image)
        
        # Weather augmentation
        if self.weather_augmentation and random.random() < 0.3:
            weather_type = random.choice(list(self.weather_transforms.keys()))
            weather_transform = self.weather_transforms[weather_type]
            img_pil = weather_transform(img_pil)
        
        # Lighting augmentation
        if random.random() < 0.5:
            img_pil = self.lighting_transforms(img_pil)
        
        # Geometric augmentation
        if random.random() < 0.7:
            img_pil = self.geometric_transforms(img_pil)
        
        # Sensor augmentation
        if self.sensor_augmentation and random.random() < 0.4:
            sensor_type = random.choice(list(self.sensor_transforms.keys()))
            sensor_transform = self.sensor_transforms[sensor_type]
            img_pil = sensor_transform(img_pil)
        
        # Convert back to tensor and restore device
        return T.ToTensor()(img_pil).to(original_device)
    
    def _augment_controls(self, controls: torch.Tensor) -> torch.Tensor:
        """Augment control sequences with realistic variations"""
        if random.random() < 0.3:
            # Add small gaussian noise to controls
            noise = torch.randn_like(controls) * 0.02  # Small noise
            
            # Ensure controls stay in valid range
            controls_aug = controls + noise
            controls_aug = torch.clamp(controls_aug, -1.0, 1.0)
            
            return controls_aug
        
        return controls
    
    def _augment_state(self, state: torch.Tensor) -> torch.Tensor:
        """Augment vehicle state with sensor noise"""
        if random.random() < 0.3:
            # Add realistic sensor noise to pose and velocity
            # Pose noise (smaller)
            pose_noise = torch.randn(6, device=state.device) * 0.01  # 1cm position, 0.01 rad orientation
            # Velocity noise (larger)
            vel_noise = torch.randn(3, device=state.device) * 0.1   # 0.1 m/s velocity
            # Acceleration noise (larger)
            acc_noise = torch.randn(3, device=state.device) * 0.2   # 0.2 m/s^2 acceleration
            
            noise = torch.cat([pose_noise, vel_noise, acc_noise])
            return state + noise
        
        return state


class RainAugmentation:
    """Simulate rain effects on images"""
    
    def __init__(self, intensity_range: Tuple[float, float] = (0.3, 0.8)):
        self.intensity_range = intensity_range
    
    def __call__(self, image: Image.Image) -> Image.Image:
        img = np.array(image)
        intensity = random.uniform(*self.intensity_range)
        
        # Create rain streaks
        h, w = img.shape[:2]
        num_streaks = int(intensity * 200)
        
        for _ in range(num_streaks):
            # Random start point
            x = random.randint(0, w - 1)
            y = random.randint(0, h // 2)  # Start from upper half
            
            # Random length and angle
            length = random.randint(10, 30)
            angle = random.uniform(-10, 10)  # Mostly vertical
            
            # Calculate end point
            rad = np.radians(angle)
            end_x = int(x + length * np.sin(rad))
            end_y = int(y + length * np.cos(rad))
            
            # Draw rain streak
            color = (200 + random.randint(-20, 20),) * 3  # Light gray
            cv2.line(img, (x, y), (end_x, end_y), color, 1)
        
        # Add overall wetness effect
        img = cv2.convertScaleAbs(img, alpha=0.9, beta=10)
        
        return Image.fromarray(img)


class FogAugmentation:
    """Simulate fog effects"""
    
    def __init__(self, density_range: Tuple[float, float] = (0.2, 0.6)):
        self.density_range = density_range
    
    def __call__(self, image: Image.Image) -> Image.Image:
        img = np.array(image)
        density = random.uniform(*self.density_range)
        
        # Create fog layer
        h, w = img.shape[:2]
        fog = np.ones((h, w, 3), dtype=np.uint8) * 200
        
        # Apply Gaussian blur for realistic fog
        kernel_size = int(density * 50) | 1  # Ensure odd number
        fog = cv2.GaussianBlur(fog, (kernel_size, kernel_size), 0)
        
        # Blend fog with original image
        alpha = density
        result = cv2.addWeighted(img, 1 - alpha, fog, alpha, 0)
        
        return Image.fromarray(result)


class NightAugmentation:
    """Simulate nighttime lighting conditions"""
    
    def __call__(self, image: Image.Image) -> Image.Image:
        # Reduce brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(0.3)
        
        # Add blue tint
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(0.8)
        
        # Add noise (sensor noise in low light)
        img_array = np.array(image)
        noise = np.random.normal(0, 10, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)


class GlareAugmentation:
    """Simulate sun glare effects"""
    
    def __call__(self, image: Image.Image) -> Image.Image:
        img = np.array(image)
        h, w = img.shape[:2]
        
        # Random glare position (usually upper area)
        center_x = random.randint(w // 4, 3 * w // 4)
        center_y = random.randint(h // 6, h // 3)
        
        # Create glare gradient
        max_radius = random.randint(50, 150)
        for r in range(max_radius, 0, -2):
            alpha = (max_radius - r) / max_radius * 0.3
            color = (255, 255, 200)  # Yellowish white
            
            cv2.circle(img, (center_x, center_y), r, color, -1)
            
            # Blend with transparency
            overlay = np.zeros_like(img)
            cv2.circle(overlay, (center_x, center_y), r, color, -1)
            img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        
        return Image.fromarray(img)


class GaussianNoise:
    """Add Gaussian noise to simulate sensor noise"""
    
    def __init__(self, std_range: Tuple[float, float] = (5, 20)):
        self.std_range = std_range
    
    def __call__(self, image: Image.Image) -> Image.Image:
        img = np.array(image)
        std = random.uniform(*self.std_range)
        
        noise = np.random.normal(0, std, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)


class MotionBlur:
    """Simulate motion blur from vehicle movement"""
    
    def __init__(self, kernel_size_range: Tuple[int, int] = (3, 15)):
        self.kernel_size_range = kernel_size_range
    
    def __call__(self, image: Image.Image) -> Image.Image:
        img = np.array(image)
        kernel_size = random.choice(range(*self.kernel_size_range, 2))
        
        # Create motion blur kernel (horizontal)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int(kernel_size/2), :] = np.ones(kernel_size) / kernel_size
        
        # Apply blur
        img = cv2.filter2D(img, -1, kernel)
        
        return Image.fromarray(img)


class ChromaticAberration:
    """Simulate chromatic aberration from lens imperfections"""
    
    def __call__(self, image: Image.Image) -> Image.Image:
        img = np.array(image)
        
        # Shift color channels slightly
        h, w = img.shape[:2]
        shift_x = random.randint(-2, 2)
        shift_y = random.randint(-2, 2)
        
        # Create shifted versions
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        red_channel = cv2.warpAffine(img[:, :, 0], M, (w, h))
        
        M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
        blue_channel = cv2.warpAffine(img[:, :, 2], M, (w, h))
        
        # Reconstruct image
        result = img.copy()
        result[:, :, 0] = red_channel
        result[:, :, 2] = blue_channel
        
        return Image.fromarray(result)


class CompressionArtifacts:
    """Simulate JPEG compression artifacts"""
    
    def __init__(self, quality_range: Tuple[int, int] = (30, 80)):
        self.quality_range = quality_range
    
    def __call__(self, image: Image.Image) -> Image.Image:
        import io
        
        # Compress and decompress to simulate artifacts
        quality = random.randint(*self.quality_range)
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        return Image.open(buffer)


class LensDistortion:
    """Simulate lens distortion effects"""
    
    def __call__(self, image: Image.Image) -> Image.Image:
        img = np.array(image)
        h, w = img.shape[:2]
        
        # Create distortion map
        k1 = random.uniform(-0.0001, 0.0001)
        k2 = random.uniform(-0.00001, 0.00001)
        
        # Apply barrel/pincushion distortion
        fx, fy = w / 2, h / 2
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Normalize coordinates
        x = (map_x - fx) / fx
        y = (map_y - fy) / fy
        
        # Apply distortion
        r2 = x**2 + y**2
        distortion = 1 + k1 * r2 + k2 * r2**2
        
        map_x = (x * distortion * fx + fx).astype(np.float32)
        map_y = (y * distortion * fy + fy).astype(np.float32)
        
        # Apply distortion
        distorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        
        return Image.fromarray(distorted)


class ShadowAugmentation:
    """Add realistic shadows to images"""
    
    def __call__(self, image: Image.Image) -> Image.Image:
        img = np.array(image)
        h, w = img.shape[:2]
        
        # Random shadow parameters
        shadow_intensity = random.uniform(0.3, 0.7)
        shadow_y = random.randint(h // 2, h)
        shadow_height = random.randint(h // 4, h // 2)
        
        # Create gradient shadow
        for y in range(shadow_y, min(shadow_y + shadow_height, h)):
            alpha = shadow_intensity * (1 - (y - shadow_y) / shadow_height)
            img[y, :] = img[y, :] * (1 - alpha)
        
        return Image.fromarray(img)


class SnowAugmentation:
    """Simulate snow effects"""
    
    def __init__(self, intensity_range: Tuple[float, float] = (0.1, 0.4)):
        self.intensity_range = intensity_range
    
    def __call__(self, image: Image.Image) -> Image.Image:
        img = np.array(image)
        intensity = random.uniform(*self.intensity_range)
        
        # Add snowflakes
        h, w = img.shape[:2]
        num_flakes = int(intensity * 500)
        
        for _ in range(num_flakes):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            
            # Draw snowflake
            color = 255  # White
            cv2.circle(img, (x, y), 1, color, -1)
        
        # Add overall brightness
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=20)
        
        return Image.fromarray(img)


# Utility functions for specific augmentation strategies
def create_adverse_weather_augmentation() -> DrivingAugmentation:
    """Create augmentation focused on adverse weather conditions"""
    return DrivingAugmentation(
        weather_augmentation=True,
        traffic_augmentation=False,
        sensor_augmentation=False,
        augment_probability=0.9
    )


def create_sensor_noise_augmentation() -> DrivingAugmentation:
    """Create augmentation focused on sensor noise and artifacts"""
    return DrivingAugmentation(
        weather_augmentation=False,
        traffic_augmentation=False,
        sensor_augmentation=True,
        augment_probability=0.8
    )


def create_comprehensive_augmentation() -> DrivingAugmentation:
    """Create comprehensive augmentation for robust training"""
    return DrivingAugmentation(
        weather_augmentation=True,
        traffic_augmentation=True,
        sensor_augmentation=True,
        augment_probability=0.85
    )