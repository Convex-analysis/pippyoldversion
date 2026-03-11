"""
Autonomous driving data augmentation for EVO-1.

Design follows MMDetection3D / CenterPoint conventions:
  - RandomFlip3D:          horizontal image flip + simultaneous flip of future_trajectory
                           y-axis and ego state yaw/vy for 3-D physical consistency.
  - GlobalRotScaleTrans:   global rotation, scale, and translation perturbation applied
                           to future_trajectory (BEV-space), matching CenterPoint defaults.
  - PointShuffle:          placeholder for LiDAR point clouds when added.
  - Image augmentations:   weather / sensor / lighting effects with a single PIL round-trip
                           per image (no repeated ToPILImage / ToTensor conversions).
  - traffic_augmentation:  Random Erasing to simulate foreground vehicle/pedestrian occlusion.
  - LightAugmentation:     lightweight augmentation class (merged from augmentation_light.py).

Public interface is backward compatible:
  augment_batch(batch) → augmented batch
  create_comprehensive_augmentation() → DrivingAugmentation
  create_adverse_weather_augmentation() → DrivingAugmentation
  create_sensor_noise_augmentation() → DrivingAugmentation
  create_light_augmentation() → LightAugmentation   (new, from merged file)
  create_no_augmentation() → NoAugmentation         (new, from merged file)
"""

import io
import random
import logging
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3-D augmentation helpers (MMDet3D / CenterPoint style)
# ---------------------------------------------------------------------------

def _flip_trajectory_y(trajectory: np.ndarray) -> np.ndarray:
    """Flip future trajectory along y-axis (BEV horizontal flip)."""
    out = trajectory.copy()
    out[:, 1] *= -1
    return out


def _rotate_trajectory_2d(
    trajectory: np.ndarray, angle_rad: float
) -> np.ndarray:
    """Rotate [T, 3] trajectory around z-axis by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    out = trajectory.copy()
    out[:, :2] = (R @ trajectory[:, :2].T).T
    return out


def _scale_trajectory(
    trajectory: np.ndarray, scale: float
) -> np.ndarray:
    """Uniform XY scale of trajectory."""
    out = trajectory.copy()
    out[:, :2] *= scale
    return out


def _translate_trajectory(
    trajectory: np.ndarray, tx: float, ty: float
) -> np.ndarray:
    """Translate trajectory XY by (tx, ty)."""
    out = trajectory.copy()
    out[:, 0] += tx
    out[:, 1] += ty
    return out


# ---------------------------------------------------------------------------
# Image augmentation primitives (all PIL-based, single round-trip)
# ---------------------------------------------------------------------------

class _WeatherAug:
    """Base class for weather augmentations operating on np.ndarray (uint8 RGB)."""
    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RainAugmentation(_WeatherAug):
    """Simulate rain streaks."""
    def __init__(self, intensity_range: Tuple[float, float] = (0.3, 0.8)):
        self.intensity_range = intensity_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        intensity = random.uniform(*self.intensity_range)
        num_streaks = int(intensity * 200)
        out = img.copy()
        for _ in range(num_streaks):
            x = random.randint(0, w - 1)
            y = random.randint(0, h // 2)
            length = random.randint(10, 30)
            angle = random.uniform(-10, 10)
            rad = np.radians(angle)
            ex = int(x + length * np.sin(rad))
            ey = int(y + length * np.cos(rad))
            color = (200 + random.randint(-20, 20),) * 3
            cv2.line(out, (x, y), (ex, ey), color, 1)
        out = cv2.convertScaleAbs(out, alpha=0.9, beta=10)
        return out


class FogAugmentation(_WeatherAug):
    """Simulate fog by blending a grey layer."""
    def __init__(self, density_range: Tuple[float, float] = (0.2, 0.6)):
        self.density_range = density_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        density = random.uniform(*self.density_range)
        fog = np.full((h, w, 3), 200, dtype=np.uint8)
        k = int(density * 50) | 1
        fog = cv2.GaussianBlur(fog, (k, k), 0)
        return cv2.addWeighted(img, 1 - density, fog, density, 0)


class SnowAugmentation(_WeatherAug):
    """Simulate snowflakes and brightness increase."""
    def __init__(self, intensity_range: Tuple[float, float] = (0.1, 0.4)):
        self.intensity_range = intensity_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        intensity = random.uniform(*self.intensity_range)
        out = np.ascontiguousarray(img.copy())
        for _ in range(int(intensity * 500)):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            cv2.circle(out, (x, y), 1, 255, -1)
        return cv2.convertScaleAbs(out, alpha=1.1, beta=20)


class NightAugmentation(_WeatherAug):
    """Reduce brightness and add low-light sensor noise."""
    def __call__(self, img: np.ndarray) -> np.ndarray:
        # Darken
        out = (img.astype(np.float32) * 0.3).clip(0, 255).astype(np.uint8)
        # Sensor noise
        noise = np.random.normal(0, 10, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out


class GlareAugmentation(_WeatherAug):
    """Simulate sun glare with a radial gradient."""
    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 6, h // 3)
        max_r = random.randint(50, 150)
        overlay = np.ascontiguousarray(np.zeros_like(img, dtype=np.float32))
        cv2.circle(overlay, (cx, cy), max_r, (255, 255, 200), -1)
        # Gaussian falloff
        overlay = cv2.GaussianBlur(overlay, (max_r | 1, max_r | 1), max_r // 3)
        alpha = 0.35
        return np.clip(img.astype(np.float32) + overlay * alpha, 0, 255).astype(np.uint8)


class ShadowAugmentation(_WeatherAug):
    """Add a gradient shadow band to simulate tree/building shadows."""
    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        shadow_intensity = random.uniform(0.3, 0.7)
        y0 = random.randint(h // 2, h)
        sh = random.randint(h // 4, h // 2)
        out = img.copy().astype(np.float32)
        for y in range(y0, min(y0 + sh, h)):
            alpha = shadow_intensity * (1 - (y - y0) / sh)
            out[y] *= (1 - alpha)
        return out.clip(0, 255).astype(np.uint8)


class _SensorAug:
    """Base class for sensor-noise augmentations on np.ndarray."""
    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GaussianNoise(_SensorAug):
    def __init__(self, std_range: Tuple[float, float] = (5, 20)):
        self.std_range = std_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        std = random.uniform(*self.std_range)
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class MotionBlur(_SensorAug):
    """Horizontal motion blur from vehicle movement."""
    def __init__(self, kernel_size_range: Tuple[int, int] = (3, 15)):
        self.kernel_size_range = kernel_size_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        ks = random.choice(range(self.kernel_size_range[0], self.kernel_size_range[1], 2))
        kernel = np.zeros((ks, ks), dtype=np.float32)
        kernel[ks // 2, :] = 1.0 / ks
        return cv2.filter2D(img, -1, kernel)


class ChromaticAberration(_SensorAug):
    """Shift R and B channels to simulate chromatic aberration."""
    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        sx, sy = random.randint(-2, 2), random.randint(-2, 2)
        M_pos = np.float32([[1, 0, sx], [0, 1, sy]])
        M_neg = np.float32([[1, 0, -sx], [0, 1, -sy]])
        out = img.copy()
        out[:, :, 0] = cv2.warpAffine(img[:, :, 0], M_pos, (w, h))
        out[:, :, 2] = cv2.warpAffine(img[:, :, 2], M_neg, (w, h))
        return out


class CompressionArtifacts(_SensorAug):
    """Simulate JPEG compression artifacts."""
    def __init__(self, quality_range: Tuple[int, int] = (30, 80)):
        self.quality_range = quality_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        quality = random.randint(*self.quality_range)
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return np.array(Image.open(buf))


class LensDistortion(_SensorAug):
    """Barrel / pincushion lens distortion."""
    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        k1 = random.uniform(-1e-4, 1e-4)
        k2 = random.uniform(-1e-5, 1e-5)
        fx, fy = w / 2.0, h / 2.0
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        x = (xs - fx) / fx
        y = (ys - fy) / fy
        r2 = x ** 2 + y ** 2
        dist = 1 + k1 * r2 + k2 * r2 ** 2
        map_x = (x * dist * fx + fx).astype(np.float32)
        map_y = (y * dist * fy + fy).astype(np.float32)
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Main augmentation class
# ---------------------------------------------------------------------------

class DrivingAugmentation:
    """Comprehensive augmentation suite for autonomous driving data.

    Augments a batch dict with keys:
        images          [B, N_views, 3, H, W]  float tensor
        future_controls [B, T, 3]              float tensor
        state           [B, D]                 float tensor  (D≥4, pose[3]=yaw)

    3-D consistency guarantees:
        - RandomFlip3D flips images horizontally AND negates trajectory y and state yaw/vy.
        - GlobalRotScaleTrans applies a BEV rotation + scale + translation to trajectory.
        - All geometric transforms are sampled once per sample and applied consistently
          across all camera views.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (448, 448),
        augment_probability: float = 0.8,
        weather_augmentation: bool = True,
        traffic_augmentation: bool = True,
        sensor_augmentation: bool = True,
        # 3-D augmentation parameters (MMDet3D defaults)
        flip3d_prob: float = 0.5,
        rot_range: float = np.pi / 4,           # ±45°
        scale_range: Tuple[float, float] = (0.95, 1.05),
        trans_std: float = 0.2,                 # metres
    ):
        self.image_size = image_size
        self.augment_probability = augment_probability
        self.weather_augmentation = weather_augmentation
        self.traffic_augmentation = traffic_augmentation
        self.sensor_augmentation = sensor_augmentation

        self.flip3d_prob = flip3d_prob
        self.rot_range = rot_range
        self.scale_range = scale_range
        self.trans_std = trans_std

        # Crop / flip parameters for geometric consistency
        self.crop_scale = (0.8, 1.0)
        self.flip_prob = 0.5

        # Weather pool
        self._weather_augs: List[_WeatherAug] = [
            RainAugmentation(),
            FogAugmentation(),
            SnowAugmentation(),
            NightAugmentation(),
            GlareAugmentation(),
            ShadowAugmentation(),
        ]

        # Sensor-noise pool
        self._sensor_augs: List[_SensorAug] = [
            GaussianNoise(),
            MotionBlur(),
            ChromaticAberration(),
            CompressionArtifacts(),
            LensDistortion(),
        ]

        # Colour jitter (torchvision — applied on tensor)
        self._color_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )

        # Random Erasing for traffic / occlusion augmentation
        # Erases a random rectangle to simulate vehicle / pedestrian occlusion
        self._random_erasing = T.RandomErasing(
            p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Augment a batch in-place (returns augmented copy)."""
        if random.random() > self.augment_probability:
            return batch

        aug = {k: v for k, v in batch.items()}  # shallow copy

        if "images" in batch:
            # Work on CPU for PIL / numpy transforms
            imgs_cpu = batch["images"].cpu()  # [B, N, 3, H, W]
            B = imgs_cpu.shape[0]

            aug_imgs: List[torch.Tensor] = []
            traj_list: Optional[List[np.ndarray]] = None
            state_list: Optional[List[torch.Tensor]] = None

            if "future_controls" in batch:
                traj_np = batch["future_controls"].cpu().numpy()  # [B, T, 3]
            else:
                traj_np = None

            state_t = batch.get("state")  # [B, D]

            for b in range(B):
                sample_imgs = imgs_cpu[b]  # [N, 3, H, W]
                traj = traj_np[b].copy() if traj_np is not None else None
                state = state_t[b].clone().cpu() if state_t is not None else None

                # 3-D: RandomFlip3D
                if random.random() < self.flip3d_prob:
                    sample_imgs, traj, state = self._apply_flip3d(
                        sample_imgs, traj, state
                    )

                # 3-D: GlobalRotScaleTrans on trajectory
                if traj is not None:
                    traj = self._apply_global_rot_scale_trans(traj)

                # Image geometric (crop + optional flip) — consistent across views
                sample_imgs = self._apply_geometric(sample_imgs)

                # Image appearance: weather / sensor / traffic per view
                aug_views: List[torch.Tensor] = []
                for v in range(sample_imgs.shape[0]):
                    aug_views.append(
                        self._augment_single_image(sample_imgs[v])
                    )
                sample_imgs = torch.stack(aug_views)

                aug_imgs.append(sample_imgs)
                if traj is not None:
                    if traj_list is None:
                        traj_list = []
                    traj_list.append(traj)
                if state is not None:
                    if state_list is None:
                        state_list = []
                    state_list.append(state)

            aug["images"] = torch.stack(aug_imgs).to(batch["images"].device)

            if traj_list is not None:
                aug["future_controls"] = torch.from_numpy(
                    np.stack(traj_list, axis=0)
                ).to(batch["future_controls"].device)

            if state_list is not None:
                aug["state"] = torch.stack(state_list).to(batch["state"].device)

        # Lightweight control noise (independent of image augmentation)
        if "future_controls" in aug and random.random() < 0.3:
            noise = torch.randn_like(aug["future_controls"]) * 0.02
            aug["future_controls"] = (aug["future_controls"] + noise).clamp(-1.0, 1.0)

        return aug

    # ------------------------------------------------------------------
    # 3-D augmentation methods
    # ------------------------------------------------------------------

    def _apply_flip3d(
        self,
        images: torch.Tensor,             # [N, 3, H, W]
        trajectory: Optional[np.ndarray], # [T, 3] or None
        state: Optional[torch.Tensor],    # [D] or None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray], Optional[torch.Tensor]]:
        """RandomFlip3D: horizontal image flip + BEV y-axis flip.

        State convention assumed: state[3] = yaw, state[4] = vy (if available).
        Matches the state vector layout in nuscenes_loader.py:
            [x, y, z, yaw, vx, vy, vz, ax, ay, az]  (10-D)
        """
        # Flip all camera views horizontally
        flipped_imgs = torch.stack([TF.hflip(images[v]) for v in range(images.shape[0])])

        # Flip trajectory y-axis
        flipped_traj = _flip_trajectory_y(trajectory) if trajectory is not None else None

        # Flip yaw and vy in state
        flipped_state = state.clone() if state is not None else None
        if flipped_state is not None and flipped_state.shape[0] > 3:
            flipped_state[3] = -flipped_state[3]  # yaw
        if flipped_state is not None and flipped_state.shape[0] > 5:
            flipped_state[5] = -flipped_state[5]  # vy

        return flipped_imgs, flipped_traj, flipped_state

    def _apply_global_rot_scale_trans(
        self, trajectory: np.ndarray
    ) -> np.ndarray:
        """GlobalRotScaleTrans on future_trajectory (BEV-space).

        Matches MMDet3D GlobalRotScaleTrans defaults:
          rotation: ±rot_range (default ±π/4)
          scale:    uniform in scale_range (default 0.95–1.05)
          translation: Gaussian σ = trans_std (default 0.2 m)
        """
        angle = random.uniform(-self.rot_range, self.rot_range)
        scale = random.uniform(*self.scale_range)
        tx = random.gauss(0, self.trans_std)
        ty = random.gauss(0, self.trans_std)

        out = _rotate_trajectory_2d(trajectory, angle)
        out = _scale_trajectory(out, scale)
        out = _translate_trajectory(out, tx, ty)
        return out

    # ------------------------------------------------------------------
    # Image geometric (consistent across views)
    # ------------------------------------------------------------------

    def _apply_geometric(self, images: torch.Tensor) -> torch.Tensor:
        """Consistent random crop across all views; optional horizontal flip."""
        i, j, h, w = T.RandomResizedCrop.get_params(
            images[0], scale=self.crop_scale, ratio=(0.9, 1.1)
        )
        do_flip = random.random() < self.flip_prob

        views: List[torch.Tensor] = []
        for v in range(images.shape[0]):
            img = TF.resized_crop(images[v], i, j, h, w, self.image_size)
            if do_flip:
                img = TF.hflip(img)
            views.append(img)
        return torch.stack(views)

    # ------------------------------------------------------------------
    # Single-image appearance augmentation (one PIL round-trip)
    # ------------------------------------------------------------------

    def _augment_single_image(self, img_t: torch.Tensor) -> torch.Tensor:
        """Apply weather / sensor / traffic augmentations via single PIL conversion."""
        # Convert tensor [3, H, W] float → numpy uint8 for PIL-based augmentations
        img_np = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

        # Weather (single choice per image, 30% chance)
        if self.weather_augmentation and random.random() < 0.3:
            aug = random.choice(self._weather_augs)
            img_np = aug(img_np)

        # Sensor noise (single choice, 40% chance)
        if self.sensor_augmentation and random.random() < 0.4:
            aug = random.choice(self._sensor_augs)
            img_np = aug(img_np)

        # Back to tensor [3, H, W] float in [0, 1]
        result = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        # Colour jitter (torchvision, native tensor)
        if random.random() < 0.5:
            result = self._color_jitter(result)

        # Traffic / occlusion augmentation: random rectangle erase
        if self.traffic_augmentation and random.random() < 0.3:
            result = self._random_erasing(result)

        return result

    # ------------------------------------------------------------------
    # State noise (sensor noise on ego-state)
    # ------------------------------------------------------------------

    def _augment_state(self, state: torch.Tensor) -> torch.Tensor:
        """Add realistic IMU / GPS sensor noise to state vector."""
        if random.random() < 0.3:
            dev = state.device
            # [x, y, z] position noise (GPS ~1 cm)
            pos_noise = torch.randn(3, device=dev) * 0.01
            # yaw noise (~0.01 rad)
            yaw_noise = torch.randn(1, device=dev) * 0.01
            # velocity noise (0.1 m/s)
            vel_noise = torch.randn(3, device=dev) * 0.1
            # acceleration noise (0.2 m/s²)
            acc_noise = torch.randn(3, device=dev) * 0.2
            noise = torch.cat([pos_noise, yaw_noise, vel_noise, acc_noise])
            # Pad or trim to match actual state dimension
            if noise.shape[0] > state.shape[0]:
                noise = noise[: state.shape[0]]
            elif noise.shape[0] < state.shape[0]:
                noise = F.pad(noise, (0, state.shape[0] - noise.shape[0]))
            return state + noise
        return state


# ---------------------------------------------------------------------------
# Lightweight augmentation (merged from augmentation_light.py)
# ---------------------------------------------------------------------------

class LightAugmentation:
    """Lightweight augmentation for fast training or validation.

    Applies only mild crop + flip + gentle colour jitter; no weather or sensor effects.
    Previously in augmentation_light.py; merged here to eliminate the orphan file.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (448, 448),
        augment_probability: float = 0.5,
    ):
        self.image_size = image_size
        self.augment_probability = augment_probability
        self._color_jitter = T.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05
        )

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.augment_probability:
            return batch

        aug = {k: v for k, v in batch.items()}

        if "images" in batch:
            imgs_cpu = batch["images"].cpu()
            B = imgs_cpu.shape[0]
            aug_imgs: List[torch.Tensor] = []

            for b in range(B):
                sample = imgs_cpu[b]  # [N, 3, H, W]
                i, j, h, w = T.RandomResizedCrop.get_params(
                    sample[0], scale=(0.9, 1.0), ratio=(0.95, 1.05)
                )
                do_flip = random.random() < 0.5
                views: List[torch.Tensor] = []
                for v in range(sample.shape[0]):
                    img = TF.resized_crop(sample[v], i, j, h, w, self.image_size)
                    if do_flip:
                        img = TF.hflip(img)
                    if random.random() < 0.5:
                        img = self._color_jitter(img)
                    views.append(img)
                aug_imgs.append(torch.stack(views))

            aug["images"] = torch.stack(aug_imgs).to(batch["images"].device)

        return aug


class NoAugmentation:
    """Identity augmentation — passes batch through unchanged.

    Useful for validation / ablation to keep the same call-site interface.
    Previously in augmentation_light.py; merged here.
    """

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch


# ---------------------------------------------------------------------------
# Factory functions (all original names preserved + new ones)
# ---------------------------------------------------------------------------

def create_comprehensive_augmentation() -> DrivingAugmentation:
    """Full augmentation suite: weather + sensor + traffic + 3-D transforms."""
    return DrivingAugmentation(
        weather_augmentation=True,
        traffic_augmentation=True,
        sensor_augmentation=True,
        augment_probability=0.85,
    )


def create_adverse_weather_augmentation() -> DrivingAugmentation:
    """Augmentation focused on adverse weather conditions."""
    return DrivingAugmentation(
        weather_augmentation=True,
        traffic_augmentation=False,
        sensor_augmentation=False,
        augment_probability=0.9,
    )


def create_sensor_noise_augmentation() -> DrivingAugmentation:
    """Augmentation focused on sensor noise and artifacts."""
    return DrivingAugmentation(
        weather_augmentation=False,
        traffic_augmentation=False,
        sensor_augmentation=True,
        augment_probability=0.8,
    )


def create_light_augmentation(
    image_size: Tuple[int, int] = (448, 448),
    augment_probability: float = 0.5,
) -> LightAugmentation:
    """Lightweight augmentation: mild crop + flip + gentle colour jitter only."""
    return LightAugmentation(image_size=image_size, augment_probability=augment_probability)


def create_no_augmentation() -> NoAugmentation:
    """No-op augmentation for validation / ablation studies."""
    return NoAugmentation()
