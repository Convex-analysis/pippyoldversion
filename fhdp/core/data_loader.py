"""
Data loading utilities for FHDP pipeline training.

Provides unified data loading interface for CIFAR-10, ImageNet, and TinyImageNet datasets.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import urllib.request
import zipfile
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import transforms


DEFAULT_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
DEFAULT_CIFAR10_STD = (0.2023, 0.1994, 0.2010)
DEFAULT_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_TINY_IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
DEFAULT_TINY_IMAGENET_DIR = "tiny-imagenet-200"


def _download_file(url: str, dest_path: str) -> None:
    """Download file from URL to destination path"""
    tmp_path = dest_path + ".tmp"
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as handle:
        shutil.copyfileobj(response, handle)
    os.replace(tmp_path, dest_path)


def _extract_archive(archive_path: str, dest_dir: str) -> None:
    """Extract zip or tar archive to destination directory"""
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(dest_dir)
        return
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(dest_dir)
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def _prepare_tiny_imagenet(
    data_dir: str,
    url: str,
    folder_name: str
) -> str:
    """Download and extract Tiny ImageNet dataset if not already present"""
    dataset_root = os.path.join(data_dir, folder_name)
    train_dir = os.path.join(dataset_root, "train")

    if os.path.isdir(train_dir):
        return dataset_root

    if not url:
        raise ValueError(
            "Tiny ImageNet url is required. Use --tiny-imagenet-url to specify a mirror."
        )

    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.basename(url.split("?")[0]) or "tiny-imagenet-200.zip"
    archive_path = os.path.join(data_dir, filename)

    if not os.path.exists(archive_path):
        print(f"[Data] Downloading Tiny ImageNet from {url} -> {archive_path}")
        _download_file(url, archive_path)
    else:
        print(f"[Data] Using existing archive: {archive_path}")

    print(f"[Data] Extracting Tiny ImageNet: {archive_path}")
    _extract_archive(archive_path, data_dir)

    default_root = os.path.join(data_dir, DEFAULT_TINY_IMAGENET_DIR)
    if (
        folder_name != DEFAULT_TINY_IMAGENET_DIR
        and os.path.isdir(default_root)
        and not os.path.isdir(dataset_root)
    ):
        os.rename(default_root, dataset_root)

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Tiny ImageNet dataset not found at {train_dir}. "
            "Expected a folder with train/val subdirectories."
        )

    return dataset_root


def build_cifar10_loader(
    batch_size: int,
    image_size: int,
    num_batches: int = 1,
    data_dir: str = "./data",
    download: bool = False,
    train: bool = True,
    mean: tuple = DEFAULT_CIFAR10_MEAN,
    std: tuple = DEFAULT_CIFAR10_STD
) -> DataLoader:
    """Build CIFAR-10 data loader with specified transforms"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )

    total_samples = batch_size * num_batches
    if total_samples < len(dataset):
        dataset = Subset(dataset, list(range(total_samples)))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def build_imagenet_loader(
    batch_size: int,
    image_size: int,
    num_batches: int = 1,
    data_dir: str = "./data",
    mean: tuple = DEFAULT_IMAGENET_MEAN,
    std: tuple = DEFAULT_IMAGENET_STD
) -> DataLoader:
    """Build ImageNet data loader with specified transforms"""
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dir = os.path.join(data_dir, "train")
    dataset_root = train_dir if os.path.isdir(train_dir) else data_dir

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"ImageNet dataset not found at {dataset_root}. "
            "Expected a folder with train/val subdirectories or a train directory."
        )

    dataset = ImageFolder(root=dataset_root, transform=transform)

    total_samples = batch_size * num_batches
    if total_samples < len(dataset):
        dataset = Subset(dataset, list(range(total_samples)))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def build_tiny_imagenet_loader(
    batch_size: int,
    image_size: int,
    num_batches: int = 1,
    data_dir: str = "./data",
    url: str = DEFAULT_TINY_IMAGENET_URL,
    folder_name: str = DEFAULT_TINY_IMAGENET_DIR,
    mean: tuple = DEFAULT_IMAGENET_MEAN,
    std: tuple = DEFAULT_IMAGENET_STD
) -> DataLoader:
    """Build Tiny ImageNet data loader with specified transforms"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset_root = _prepare_tiny_imagenet(data_dir, url, folder_name)
    train_root = os.path.join(dataset_root, "train")
    dataset = ImageFolder(root=train_root, transform=transform)

    total_samples = batch_size * num_batches
    if total_samples < len(dataset):
        dataset = Subset(dataset, list(range(total_samples)))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def build_data_loader(
    dataset_name: str,
    batch_size: int,
    image_size: int,
    num_batches: int = 1,
    data_dir: str = "./data",
    download: bool = False,
    tiny_imagenet_url: str = DEFAULT_TINY_IMAGENET_URL,
    tiny_imagenet_dir: str = DEFAULT_TINY_IMAGENET_DIR
) -> DataLoader:
    """Build data loader based on dataset name (cifar10, imagenet, tinyimagenet)"""
    dataset_key = (dataset_name or "cifar10").lower().replace("-", "")

    if dataset_key == "imagenet":
        return build_imagenet_loader(
            batch_size, image_size, num_batches, data_dir
        )

    if dataset_key == "tinyimagenet":
        return build_tiny_imagenet_loader(
            batch_size,
            image_size,
            num_batches,
            data_dir,
            url=tiny_imagenet_url,
            folder_name=tiny_imagenet_dir
        )

    return build_cifar10_loader(
        batch_size, image_size, num_batches, data_dir, download
    )


def build_eval_loader(
    dataset_name: str,
    batch_size: int,
    image_size: int,
    num_batches: int = 1,
    data_dir: str = "./data",
    download: bool = False
) -> Optional[DataLoader]:
    """Build evaluation data loader (only CIFAR-10 supported for eval)"""
    dataset_key = (dataset_name or "cifar10").lower().replace("-", "")

    if dataset_key != "cifar10":
        print(f"[Eval] Dataset {dataset_name} not supported for eval, skip")
        return None

    return build_cifar10_loader(
        batch_size, image_size, num_batches, data_dir, download, train=False
    )