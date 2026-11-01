"""
IPA Deepfake Detection - Dataset Preprocessing & DataLoader Script
==================================================================
Member A: Dataset Preparation, Preprocessing, and FFT Analysis

Description:
    This script provides a reusable pipeline for loading and preprocessing
    the deepfake image dataset using PyTorch and Torchvision.

Dataset Directory Structure Expected:
    data/
    └── subset/
        ├── train/
        │   ├── fake/
        │   └── real/
        ├── valid/
        │   ├── fake/
        │   └── real/
        └── test/
            ├── fake/
            └── real/

Preprocessing Steps:
    1. Resize images to 128 x 128 pixels.
    2. Convert PIL images to PyTorch tensors (values in [0.0, 1.0]).
    3. Normalize tensors using ImageNet mean [0.485, 0.456, 0.406] and
       standard deviation [0.229, 0.224, 0.225].

Class Mapping:
    - fake: 0
    - real: 1
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Global Constants & Defaults
# ---------------------------------------------------------------------------
IMAGE_SIZE: Tuple[int, int] = (128, 128)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_BATCH_SIZE: int = 32
EXPECTED_CLASS_MAPPING: Dict[str, int] = {"fake": 0, "real": 1}

# Reliable project root resolution based on this file's absolute path
# File location: <PROJECT_ROOT>/ml/scripts/preprocess.py -> parents[2] is <PROJECT_ROOT>
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "subset"


def get_data_transforms() -> transforms.Compose:
    """
    Returns torchvision transforms for deepfake image preprocessing:
      - Resize image to 128x128
      - Convert PIL image to Tensor
      - Normalize with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def validate_split_directory(split_path: Path, split_name: str) -> None:
    """
    Validates that a split directory exists and contains the required
    'fake' and 'real' subdirectories.

    Raises:
        FileNotFoundError: If the split directory or required class subdirectories are missing.
    """
    if not split_path.exists() or not split_path.is_dir():
        raise FileNotFoundError(
            f"Missing '{split_name}' directory at: {split_path}\n"
            f"Please ensure '{split_name}' exists inside the dataset folder."
        )

    for class_name in EXPECTED_CLASS_MAPPING.keys():
        class_dir = split_path / class_name
        if not class_dir.exists() or not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class folder '{class_name}' in '{split_name}' split: {class_dir}\n"
                f"Expected structure:\n"
                f"  {split_path}/\n"
                f"    ├── fake/\n"
                f"    └── real/"
            )


def load_dataset_split(split_path: Path, split_name: str, transform: transforms.Compose) -> datasets.ImageFolder:
    """
    Loads a single dataset split using torchvision.datasets.ImageFolder
    and verifies that the class mapping strictly matches {'fake': 0, 'real': 1}.

    Args:
        split_path: Path to the split directory (e.g., data/subset/train).
        split_name: Name of the split ('train', 'valid', or 'test').
        transform: Torchvision transformation pipeline.

    Returns:
        datasets.ImageFolder: Loaded dataset split.
    """
    validate_split_directory(split_path, split_name)

    dataset = datasets.ImageFolder(root=str(split_path), transform=transform)

    # Verify that the class mapping strictly matches expected indices
    if dataset.class_to_idx != EXPECTED_CLASS_MAPPING:
        raise ValueError(
            f"Class mapping mismatch in '{split_name}' split!\n"
            f"  Expected: {EXPECTED_CLASS_MAPPING}\n"
            f"  Found:    {dataset.class_to_idx}\n"
            f"Ensure subfolders are named exactly 'fake' and 'real'."
        )

    return dataset


def get_dataloaders(
    data_dir: Optional[Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[Dict[str, datasets.ImageFolder], Dict[str, DataLoader]]:
    """
    Builds and returns ImageFolder datasets and DataLoaders for train, valid, and test splits.

    Args:
        data_dir: Path to the dataset root folder containing train/, valid/, test/.
                  Defaults to `<PROJECT_ROOT>/data/subset`.
        batch_size: Number of samples per batch (default: 32).
        num_workers: Number of subprocesses for data loading (default: 0 for compatibility).
        pin_memory: If True, pin memory when loading to GPU (default: False).

    Returns:
        Tuple containing:
            - datasets_dict: Dictionary with keys 'train', 'valid', 'test' mapping to ImageFolder.
            - dataloaders_dict: Dictionary with keys 'train', 'valid', 'test' mapping to DataLoader.
              Only the 'train' DataLoader has shuffle=True.
    """
    if batch_size <= 0:
        raise ValueError(
            f"Invalid batch_size: {batch_size}. Batch size must be a positive integer (greater than 0)."
        )

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    else:
        data_dir = Path(data_dir).resolve()

    # Verify that root data directory exists
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found at: {data_dir}\n"
            f"Expected directory structure:\n"
            f"{data_dir}/\n"
            f"├── train/\n"
            f"│   ├── fake/\n"
            f"│   └── real/\n"
            f"├── valid/\n"
            f"│   ├── fake/\n"
            f"│   └── real/\n"
            f"└── test/\n"
            f"    ├── fake/\n"
            f"    └── real/"
        )

    # Get preprocessing transformations
    data_transform = get_data_transforms()

    # Load datasets for each split
    split_names = ["train", "valid", "test"]
    dataset_splits: Dict[str, datasets.ImageFolder] = {}

    for split in split_names:
        split_path = data_dir / split
        dataset_splits[split] = load_dataset_split(split_path, split, data_transform)

    # Create DataLoaders
    # Note: Only the training DataLoader must be shuffled
    dataloaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            dataset_splits["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "valid": DataLoader(
            dataset_splits["valid"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            dataset_splits["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    return dataset_splits, dataloaders


def print_summary(
    datasets_dict: Dict[str, datasets.ImageFolder],
    dataloaders_dict: Dict[str, DataLoader],
) -> None:
    """
    Prints preprocessing summary including:
      - Class mapping
      - Number of images in each split
      - Batch image shape
      - Batch label shape
    """
    print("=" * 60)
    print("IPA Deepfake Detection - Dataset Preprocessing Summary")
    print("=" * 60)

    # 1. Print class mapping
    class_mapping = datasets_dict["train"].class_to_idx
    print(f"\n[+] Class Mapping: {class_mapping}")
    for class_name, class_idx in class_mapping.items():
        print(f"    - {class_name} -> {class_idx}")

    # 2. Print image counts in each split
    print("\n[+] Dataset Splits & Image Counts:")
    for split_name, dataset in datasets_dict.items():
        print(f"    - {split_name.capitalize():<5} split: {len(dataset):>5} images")

    # 3. Print batch image & label shape from training DataLoader
    print("\n[+] Batch Tensor Inspection:")
    train_loader = dataloaders_dict["train"]
    if len(datasets_dict["train"]) == 0:
        print("    [!] Training dataset is empty. Cannot inspect batch shapes.")
    else:
        try:
            sample_images, sample_labels = next(iter(train_loader))
            print(f"    - Batch image shape : {list(sample_images.shape)} (format: [Batch, Channels, Height, Width])")
            print(f"    - Batch label shape : {list(sample_labels.shape)} (format: [Batch])")
            print(f"    - Image tensor dtype: {sample_images.dtype}")
            print(f"    - Label tensor dtype: {sample_labels.dtype}")
        except Exception as e:
            print(f"    [!] Error inspecting batch: {e}")

    print("\n" + "=" * 60)
    print("Preprocessing pipeline loaded successfully.")
    print("=" * 60)


def main() -> None:
    """
    Main entry point for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="IPA Deepfake Detection - Dataset Preprocessing & DataLoader"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Path to dataset subset directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"DataLoader batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    data_dir_path = Path(args.data_dir)

    try:
        datasets_dict, dataloaders_dict = get_dataloaders(
            data_dir=data_dir_path,
            batch_size=args.batch_size,
        )
        print_summary(datasets_dict, dataloaders_dict)
    except FileNotFoundError as err:
        print(f"\n[ERROR] Directory structure validation failed:\n{err}", file=sys.stderr)
        sys.exit(1)
    except ValueError as err:
        print(f"\n[ERROR] Invalid parameter:\n{err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"\n[ERROR] An unexpected error occurred:\n{err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
