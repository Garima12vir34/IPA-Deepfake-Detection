"""
IPA Deepfake Detection - Fast Fourier Transform (FFT) Analysis
==============================================================
Member A: Dataset Preparation, Preprocessing, and FFT Analysis

Description:
    This script extracts frequency-domain features from image data using
    the 2D Fast Fourier Transform (FFT). It evaluates the spectral energy
    distribution across low, medium, and high radial frequency bands
    for real vs. fake images to explore frequency artifacts.

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

Labels:
    - fake = 0
    - real = 1

Pipeline:
    1. Sample a reproducible subset (100 real, 100 fake) from train/ ONLY.
    2. Convert images to single-channel grayscale and resize to 128x128.
    3. Compute 2D FFT (np.fft.fft2) and center DC component with np.fft.fftshift.
    4. Compute energy spectrum as squared magnitude (|F(u, v)|^2).
    5. Partition spectrum into 3 radial frequency bands:
       - Low frequency  : radius <= 16
       - Mid frequency  : 16 < radius <= 32
       - High frequency : radius > 32
    6. Compute normalized energy ratio for each band.
    7. Generate 5 output artifacts:
       - results/fft/individual_fft_features.csv
       - results/fft/frequency_summary.csv
       - results/fft/fft_feature_boxplots.png
       - results/fft/frequency_comparison.png
       - docs/fft_analysis_report.txt
"""

import sys
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Global Constants & Defaults
# ---------------------------------------------------------------------------
IMAGE_SIZE: Tuple[int, int] = (128, 128)
DEFAULT_SAMPLE_SIZE: int = 100
DEFAULT_SEED: int = 42

# Label mapping
CLASS_MAPPING: Dict[str, int] = {"fake": 0, "real": 1}

# Frequency radius thresholds (for 128x128 grid, center at (64, 64))
LOW_FREQ_RADIUS: float = 16.0
MID_FREQ_RADIUS: float = 32.0

# Supported image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Reliable project root resolution based on this file's location:
# <PROJECT_ROOT>/ml/scripts/fft_analysis.py -> parents[2] is <PROJECT_ROOT>
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "subset"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "fft"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs"


def compute_radial_masks(
    height: int = 128,
    width: int = 128,
    r_low: float = LOW_FREQ_RADIUS,
    r_mid: float = MID_FREQ_RADIUS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes boolean masks for low, mid, and high radial frequency bands
    centered at (height // 2, width // 2).

    Args:
        height: Image height in pixels (default: 128).
        width: Image width in pixels (default: 128).
        r_low: Outer radius cutoff for low frequencies (default: 16.0).
        r_mid: Outer radius cutoff for mid frequencies (default: 32.0).

    Returns:
        Tuple of (low_mask, mid_mask, high_mask) as boolean ndarrays.
    """
    center_y, center_x = height // 2, width // 2
    y_coords, x_coords = np.ogrid[:height, :width]
    radius = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)

    low_mask = radius <= r_low
    mid_mask = (radius > r_low) & (radius <= r_mid)
    high_mask = radius > r_mid

    return low_mask, mid_mask, high_mask


def compute_fft_energy_ratios(
    image_path: Path,
    low_mask: np.ndarray,
    mid_mask: np.ndarray,
    high_mask: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """
    Loads an image, converts to grayscale, resizes to 128x128, computes the 2D FFT,
    shifts the spectrum, and calculates the normalized energy ratio for
    low, mid, and high frequency bands.

    Args:
        image_path: Path to the image file.
        low_mask: Boolean mask for low frequencies.
        mid_mask: Boolean mask for mid frequencies.
        high_mask: Boolean mask for high frequencies.

    Returns:
        Tuple of (low_ratio, mid_ratio, high_ratio) as floats, or None if reading fails.
    """
    try:
        resample_mode = (
            Image.Resampling.BILINEAR
            if hasattr(Image, "Resampling")
            else Image.BILINEAR
        )
        with Image.open(image_path) as img:
            # Step 1: Grayscale conversion and resize to 128x128
            gray_img = img.convert("L").resize(IMAGE_SIZE, resample=resample_mode)
            img_arr = np.asarray(gray_img, dtype=np.float64)

        # Step 2: 2D FFT using NumPy
        fft2 = np.fft.fft2(img_arr)

        # Step 3: Shift DC component to the center of spectrum
        fft_shifted = np.fft.fftshift(fft2)

        # Step 4: Calculate energy as squared magnitude |F(u, v)|^2
        energy_spectrum = np.abs(fft_shifted) ** 2
        total_energy = np.sum(energy_spectrum)

        if total_energy <= 0.0:
            return (0.0, 0.0, 0.0)

        # Steps 5 & 6: Band energy and normalized ratios
        low_energy = np.sum(energy_spectrum[low_mask])
        mid_energy = np.sum(energy_spectrum[mid_mask])
        high_energy = np.sum(energy_spectrum[high_mask])

        low_ratio = float(low_energy / total_energy)
        mid_ratio = float(mid_energy / total_energy)
        high_ratio = float(high_energy / total_energy)

        return (low_ratio, mid_ratio, high_ratio)

    except Exception:
        return None


def get_sorted_image_files(folder: Path) -> List[Path]:
    """
    Returns a deterministically sorted list of valid image file paths in a folder.
    """
    if not folder.is_dir():
        return []
    return sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ])


def validate_training_directories(data_dir: Path, sample_size: int) -> Tuple[Path, Path]:
    """
    Validates that the dataset structure exists and contains enough training images
    to sample without evaluation split leakage.

    Raises:
        FileNotFoundError: If directories are missing.
        ValueError: If there are fewer images than the requested sample size.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found at: {data_dir}\n"
            f"Expected directory structure:\n"
            f"  {data_dir}/\n"
            f"  └── train/\n"
            f"      ├── fake/\n"
            f"      └── real/"
        )

    train_dir = data_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training split directory missing: {train_dir}")

    train_real = train_dir / "real"
    train_fake = train_dir / "fake"

    if not train_real.is_dir():
        raise FileNotFoundError(f"Real training folder missing: {train_real}")
    if not train_fake.is_dir():
        raise FileNotFoundError(f"Fake training folder missing: {train_fake}")

    real_files = get_sorted_image_files(train_real)
    fake_files = get_sorted_image_files(train_fake)

    if len(real_files) < sample_size:
        raise ValueError(
            f"Insufficient real training images in {train_real}.\n"
            f"Found: {len(real_files)}, Required: {sample_size}"
        )
    if len(fake_files) < sample_size:
        raise ValueError(
            f"Insufficient fake training images in {train_fake}.\n"
            f"Found: {len(fake_files)}, Required: {sample_size}"
        )

    return train_real, train_fake


def extract_class_features(
    files: List[Path],
    label_name: str,
    sample_size: int,
    seed: int,
    low_mask: np.ndarray,
    mid_mask: np.ndarray,
    high_mask: np.ndarray,
    skipped_files: List[Tuple[str, str]],
) -> List[Dict]:
    """
    Selects a reproducible sample of images from the training split and
    computes radial FFT features, skipping and reporting corrupted files.
    """
    rng = random.Random(seed)
    # Reproducible sampling from sorted files
    sampled_files = rng.sample(files, sample_size)
    remaining_files = [f for f in files if f not in set(sampled_files)]
    replacement_iter = iter(remaining_files)

    records: List[Dict] = []

    for img_path in sampled_files:
        current_target = img_path
        features = compute_fft_energy_ratios(current_target, low_mask, mid_mask, high_mask)

        # Handle unreadable/corrupted images by reporting and replacing if available
        while features is None:
            skipped_files.append((str(current_target), "Corrupted or unreadable image file"))
            print(f"[WARNING] Skipping unreadable image '{current_target.name}'", file=sys.stderr)
            try:
                current_target = next(replacement_iter)
                features = compute_fft_energy_ratios(current_target, low_mask, mid_mask, high_mask)
            except StopIteration:
                break

        if features is not None:
            low_ratio, mid_ratio, high_ratio = features
            records.append({
                "filename": current_target.name,
                "label": label_name,
                "low_frequency": low_ratio,
                "mid_frequency": mid_ratio,
                "high_frequency": high_ratio,
            })

    if len(records) < sample_size:
        raise RuntimeError(
            f"Failed to extract the requested {sample_size} valid images for class '{label_name}'.\n"
            f"Only {len(records)}/{sample_size} valid images could be processed successfully.\n"
            f"Encountered {len(skipped_files)} corrupted/unreadable image file(s).\n"
            f"Execution halted to prevent generating incomplete analysis results."
        )

    return records


def generate_feature_boxplots(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generates a 1x3 boxplot chart comparing low, mid, and high frequency
    energy distributions between Real and Fake images.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    bands = [
        ("low_frequency", "Low Frequency (radius \u2264 16)"),
        ("mid_frequency", "Mid Frequency (16 < radius \u2264 32)"),
        ("high_frequency", "High Frequency (radius > 32)"),
    ]

    real_df = df[df["label"] == "real"]
    fake_df = df[df["label"] == "fake"]

    for ax, (col_name, title) in zip(axes, bands):
        data = [real_df[col_name].dropna().values, fake_df[col_name].dropna().values]
        box = ax.boxplot(
            data,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 5,
            },
        )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Real", "Fake"])
        colors = ["#4C72B0", "#DD8452"]
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized Energy Ratio", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Radial FFT Energy Distributions (Real vs. Fake)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_frequency_comparison(
    summary_df: pd.DataFrame,
    real_stds: List[float],
    fake_stds: List[float],
    output_path: Path,
) -> None:
    """
    Generates a grouped bar chart of mean radial frequency energy ratios
    with standard deviation error bars for Real vs Fake classes.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    categories = [
        "Low Frequency\n(radius \u2264 16)",
        "Mid Frequency\n(16 < radius \u2264 32)",
        "High Frequency\n(radius > 32)",
    ]
    x_pos = np.arange(len(categories))
    bar_width = 0.35

    real_means = summary_df["Real_Mean"].tolist()
    fake_means = summary_df["Fake_Mean"].tolist()

    rects1 = ax.bar(
        x_pos - bar_width / 2,
        real_means,
        bar_width,
        yerr=real_stds,
        capsize=5,
        label="Real",
        color="#4C72B0",
        alpha=0.85,
    )
    rects2 = ax.bar(
        x_pos + bar_width / 2,
        fake_means,
        bar_width,
        yerr=fake_stds,
        capsize=5,
        label="Fake",
        color="#DD8452",
        alpha=0.85,
    )

    ax.set_ylabel("Mean Normalized Energy Ratio", fontsize=11, fontweight="bold")
    ax.set_title("Frequency Band Energy Comparison: Real vs. Fake Images", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Annotate bar values
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_analysis_report(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    real_stds: List[float],
    fake_stds: List[float],
    skipped_files: List[Tuple[str, str]],
    output_path: Path,
) -> str:
    """
    Creates the text analysis report using the exact same DataFrame statistics,
    documenting methodology, summary statistics, and exploratory limitations.
    """
    real_count = len(df[df["label"] == "real"])
    fake_count = len(df[df["label"] == "fake"])
    total_count = real_count + fake_count

    real_means = summary_df["Real_Mean"].tolist()
    fake_means = summary_df["Fake_Mean"].tolist()

    report_lines = [
        "FFT ANALYSIS REPORT",
        "===================",
        "",
        f"Dataset: {total_count} training images ({real_count} real, {fake_count} fake)",
        "Image preprocessing: Grayscale, resized to 128x128",
        "Method: 2D FFT followed by fftshift",
        "Energy: Squared magnitude of the FFT",
        "Frequency bands:",
        "Low: radius <= 16",
        "Mid: 16 < radius <= 32",
        "High: radius > 32",
        "",
        "MEAN AND STANDARD DEVIATION",
        "---------------------------",
        f"Low frequency - Real: {real_means[0]:.6f} (SD: {real_stds[0]:.6f}), Fake: {fake_means[0]:.6f} (SD: {fake_stds[0]:.6f})",
        f"Mid frequency - Real: {real_means[1]:.6f} (SD: {real_stds[1]:.6f}), Fake: {fake_means[1]:.6f} (SD: {fake_stds[1]:.6f})",
        f"High frequency - Real: {real_means[2]:.6f} (SD: {real_stds[2]:.6f}), Fake: {fake_means[2]:.6f} (SD: {fake_stds[2]:.6f})",
        "",
        "OBSERVATION",
        "-----------",
        "The frequency energy distributions overlap between real and fake images.",
        "These exploratory FFT features alone do not establish reliable classification.",
        "Further evaluation is required before drawing conclusions.",
    ]

    if skipped_files:
        report_lines.append("")
        report_lines.append(f"SKIPPED / CORRUPTED FILES ({len(skipped_files)}):")
        for file_path, reason in skipped_files:
            report_lines.append(f" - {file_path}: {reason}")

    report_text = "\n".join(report_lines) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return report_text


def run_fft_pipeline(
    data_dir: Path,
    results_dir: Path,
    docs_dir: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> None:
    """
    Executes the complete FFT feature extraction and reporting pipeline.
    """
    print("=" * 65)
    print("IPA Deepfake Detection - FFT Frequency Analysis")
    print("=" * 65)

    # 1. Validate directories and sample availability
    train_real, train_fake = validate_training_directories(data_dir, sample_size)
    print(f"[+] Dataset Path         : {data_dir}")
    print(f"[+] Target Training Split : train/real & train/fake")
    print(f"[+] Target Sample Size   : {sample_size} real, {sample_size} fake (Seed: {seed})")

    # 2. Precompute radial frequency masks
    low_mask, mid_mask, high_mask = compute_radial_masks(IMAGE_SIZE[0], IMAGE_SIZE[1])

    # 3. Extract features from training split
    real_files = get_sorted_image_files(train_real)
    fake_files = get_sorted_image_files(train_fake)
    skipped_files: List[Tuple[str, str]] = []

    print("\n[+] Extracting FFT features for Real training images...")
    real_records = extract_class_features(
        real_files,
        label_name="real",
        sample_size=sample_size,
        seed=seed,
        low_mask=low_mask,
        mid_mask=mid_mask,
        high_mask=high_mask,
        skipped_files=skipped_files,
    )

    print("[+] Extracting FFT features for Fake training images...")
    fake_records = extract_class_features(
        fake_files,
        label_name="fake",
        sample_size=sample_size,
        seed=seed,
        low_mask=low_mask,
        mid_mask=mid_mask,
        high_mask=high_mask,
        skipped_files=skipped_files,
    )

    combined_records = real_records + fake_records
    if not combined_records:
        raise RuntimeError("No image features could be extracted.")

    df = pd.DataFrame(combined_records)

    # 4. Save individual features CSV
    results_dir.mkdir(parents=True, exist_ok=True)
    individual_csv_path = results_dir / "individual_fft_features.csv"
    df.to_csv(individual_csv_path, index=False)
    print(f"\n[+] Saved individual features CSV : {individual_csv_path}")

    # 5. Compute summary statistics from the exact same DataFrame
    real_df = df[df["label"] == "real"]
    fake_df = df[df["label"] == "fake"]

    real_means = [
        float(real_df["low_frequency"].mean()),
        float(real_df["mid_frequency"].mean()),
        float(real_df["high_frequency"].mean()),
    ]
    fake_means = [
        float(fake_df["low_frequency"].mean()),
        float(fake_df["mid_frequency"].mean()),
        float(fake_df["high_frequency"].mean()),
    ]
    real_stds = [
        float(real_df["low_frequency"].std()),
        float(real_df["mid_frequency"].std()),
        float(real_df["high_frequency"].std()),
    ]
    fake_stds = [
        float(fake_df["low_frequency"].std()),
        float(fake_df["mid_frequency"].std()),
        float(fake_df["high_frequency"].std()),
    ]

    summary_df = pd.DataFrame({
        "Frequency": ["Low", "Mid", "High"],
        "Real_Mean": real_means,
        "Fake_Mean": fake_means,
        "Real_Std": real_stds,
        "Fake_Std": fake_stds,
    })
    summary_csv_path = results_dir / "frequency_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[+] Saved frequency summary CSV   : {summary_csv_path}")

    # 6. Generate visualizations
    boxplots_path = results_dir / "fft_feature_boxplots.png"
    generate_feature_boxplots(df, boxplots_path)
    print(f"[+] Saved distribution boxplots   : {boxplots_path}")

    comparison_path = results_dir / "frequency_comparison.png"
    generate_frequency_comparison(summary_df, real_stds, fake_stds, comparison_path)
    print(f"[+] Saved frequency comparison bar: {comparison_path}")

    # 7. Generate exploratory report
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "fft_analysis_report.txt"
    generate_analysis_report(df, summary_df, real_stds, fake_stds, skipped_files, report_path)
    print(f"[+] Saved exploratory text report : {report_path}")

    if skipped_files:
        print(f"\n[!] Notice: {len(skipped_files)} corrupted files were skipped during extraction.")

    print("\n" + "=" * 65)
    print("FFT Analysis Pipeline Completed Successfully.")
    print("=" * 65)


def main() -> None:
    """
    Main entry point for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="IPA Deepfake Detection - FFT Frequency Analysis Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Path to dataset subset directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directory to save FFT results (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=str(DEFAULT_DOCS_DIR),
        help=f"Directory to save FFT analysis report (default: {DEFAULT_DOCS_DIR})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of images per class to sample (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for sampling reproducibility (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    if args.sample_size <= 0:
        print(
            f"\n[ERROR] Invalid sample size: {args.sample_size}. Must be a positive integer.",
            file=sys.stderr,
        )
        sys.exit(1)

    data_dir_path = Path(args.data_dir).resolve()
    results_dir_path = Path(args.results_dir).resolve()
    docs_dir_path = Path(args.docs_dir).resolve()

    try:
        run_fft_pipeline(
            data_dir=data_dir_path,
            results_dir=results_dir_path,
            docs_dir=docs_dir_path,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    except FileNotFoundError as err:
        print(f"\n[ERROR] Directory structure validation failed:\n{err}", file=sys.stderr)
        sys.exit(1)
    except ValueError as err:
        print(f"\n[ERROR] Input parameter validation failed:\n{err}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as err:
        print(f"\n[ERROR] Extraction failed:\n{err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"\n[ERROR] An unexpected error occurred:\n{err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
