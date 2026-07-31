"""
Gujarati Character Dataset Preprocessing Pipeline
--------------------------------------------------
Steps:
 1. Load raw images (raw/<label>/*.jpg)
 2. Verify images (corrupted / blurry) and copy valid ones to data/processed
 3. Build / load label map (label <-> index)
 4. Resize + color space conversion (BGR -> RGB)
 5. Pixel normalization (ImageNet mean/std) via Albumentations
 6. Data augmentation pipeline — generates augmented copies and SAVES them to disk
    (no popup windows; preview grids are saved as PNG files instead of plt.show())

Notes:
 - duplicate-detection (perceptual hashing) removed
 - train/val/test splitting removed — this script only prepares a clean,
   optionally augmented image pool
 - blur threshold lowered so mildly blurry images are kept; only clearly
   blurry images get filtered out
 - background normalization removed — images preserve original appearance
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # never open a GUI popup window; always save figures to disk
import matplotlib.pyplot as plt

import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RAW_DIR        = Path("dataset/raw")        # raw/<label>/*.jpg  (label = a Gujarati character folder)
PROCESSED_DIR  = Path("data/processed")     # verified, clean copies land here
CLEAN_DIR      = Path("data/clean")         # resized copies land here
AUG_DIR        = Path("data/augmented")     # saved augmented images land here
PREVIEW_DIR    = Path("data/previews")      # saved matplotlib preview grids (no popups)
LABEL_MAP_PATH = Path("configs/label_map.json")

IMG_SIZE   = 224
BATCH_SIZE = 32
SEED       = 42

# Lowered so mildly blurry images are still accepted — only clearly blurry
# images (very low sharpness score) get filtered out.
BLUR_THRESHOLD = 30.0   # was 100.0 — tune empirically by checking known-blurry samples

# How many augmented copies to generate per clean image
N_AUGMENTS_PER_IMAGE = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# Step 1: Load images
# --------------------------------------------------------------------------- #
def list_dataset(root: Path):
    """Scan root/<label>/*.jpg and return (filepaths, label_names) — mirrors what ImageFolder does."""
    labels = sorted([d.name for d in root.iterdir() if d.is_dir()])
    samples = []
    for label in labels:
        for img_path in (root / label).glob("*.jpg"):
            samples.append((str(img_path), label))
    return samples, labels


# --------------------------------------------------------------------------- #
# Step 2: Verify images (corrupted / blurry) — duplicate check removed
# --------------------------------------------------------------------------- #
def is_corrupted(path: str) -> bool:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img is None or img.size == 0


def blur_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_blurry(img: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    return blur_score(img) < threshold


def verify_class_folder(folder: Path):
    """Returns dict of valid/corrupted/blurry paths for one class folder."""
    report = {"valid": [], "corrupted": [], "blurry": []}

    for img_path in sorted(folder.glob("*.jpg")):
        path_str = str(img_path)

        if is_corrupted(path_str):
            report["corrupted"].append(path_str)
            continue

        img = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if is_blurry(img):
            report["blurry"].append(path_str)
            continue

        report["valid"].append(path_str)

    return report


def verify_full_dataset(raw_root: Path, output_root: Path):
    """Copies only 'valid' images into output_root/<label>/, prints a per-class QC report."""
    summary = {}
    for label_dir in sorted(raw_root.iterdir()):
        if not label_dir.is_dir():
            continue
        report = verify_class_folder(label_dir)
        summary[label_dir.name] = {k: len(v) for k, v in report.items()}

        out_dir = output_root / label_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for valid_path in report["valid"]:
            shutil.copy(valid_path, out_dir / Path(valid_path).name)

    return summary


# --------------------------------------------------------------------------- #
# Step 3: Label encoding
# --------------------------------------------------------------------------- #
def build_label_map(processed_root: Path, save_path: Path):
    labels = sorted([d.name for d in processed_root.iterdir() if d.is_dir()])
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {"label_to_idx": label_to_idx, "idx_to_label": idx_to_label},
            f, ensure_ascii=False, indent=2
        )

    print(f"Saved {len(labels)} classes to {save_path}")
    return label_to_idx, idx_to_label


def load_label_map(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    idx_to_label = {int(k): v for k, v in data["idx_to_label"].items()}  # cast keys back to int
    return data["label_to_idx"], idx_to_label


# --------------------------------------------------------------------------- #
# Step 4: Resize and color space conversion
# --------------------------------------------------------------------------- #
def resize_and_convert(img_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize with INTER_AREA (best for shrinking) then BGR -> RGB."""
    resized = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb


def build_clean_dataset(processed_root: Path, clean_root: Path):
    """
    Runs resize + RGB conversion over every valid image and SAVES each result 
    to clean_root/<label>/, so the cleaned images persist on disk.
    """
    sample_paths, labels = list_dataset(processed_root)
    for img_path, label in sample_paths:
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = resize_and_convert(bgr)

        out_dir = clean_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(img_path).name
        # convert RGB -> BGR before saving with cv2
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(sample_paths)} cleaned (resized) images to {clean_root}")
    return sample_paths, labels


# --------------------------------------------------------------------------- #
# Step 5/6: Albumentations pipelines (normalization + augmentation)
# --------------------------------------------------------------------------- #
train_transform = A.Compose([
    A.Rotate(limit=15, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), p=0.5),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
    A.GaussianBlur(blur_limit=(3, 3), p=0.2),
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.85, 1.0), p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 4),
        hole_height_range=(8, 24),
        hole_width_range=(8, 24),
        p=0.3
    ),
    A.Resize(IMG_SIZE, IMG_SIZE),
])  # NOTE: Normalize + ToTensorV2 deliberately left out here so we can save
    # augmented images as viewable jpg files; apply Normalize/ToTensorV2
    # separately at DataLoader time for training.

val_test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.numpy().transpose(1, 2, 0)
    img = (img * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)


def augment_and_save_dataset(clean_root: Path, aug_root: Path,
                              n_augments: int = N_AUGMENTS_PER_IMAGE):
    """
    For every clean image, generates n_augments augmented copies and SAVES
    them as real jpg files under aug_root/<label>/ — so augmentation output
    actually persists on disk instead of only being shown in a popup window.
    """
    sample_paths, _ = list_dataset(clean_root)
    total_saved = 0

    for img_path, label in sample_paths:
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        out_dir = aug_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(img_path).stem

        for i in range(n_augments):
            augmented = train_transform(image=rgb)["image"]  # uint8 RGB array
            out_path = out_dir / f"{stem}_aug{i}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            total_saved += 1

    print(f"Saved {total_saved} augmented images to {aug_root}")


def save_preview_grid(rgb: np.ndarray, label: str, save_path: Path):
    """Saves a preview figure to disk instead of calling plt.show() (no popups)."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(rgb)
    ax.set_title(f"Resized {IMG_SIZE}x{IMG_SIZE}, RGB")
    ax.axis("off")
    plt.suptitle(f"label: {label}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved preview to {save_path}")


def save_augmentation_preview(rgb: np.ndarray, label: str, save_path: Path, n: int = 5):
    """Saves a grid of n augmented versions to disk instead of plt.show()."""
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for ax in axes:
        augmented = train_transform(image=rgb)["image"]
        ax.imshow(augmented)
        ax.axis("off")
    plt.suptitle(f"{n} augmented versions — label: {label}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved augmentation preview to {save_path}")


# --------------------------------------------------------------------------- #
# Main pipeline (no train/val/test splitting)
# --------------------------------------------------------------------------- #
def main():
    # Step 1: scan raw data
    samples, labels = list_dataset(RAW_DIR)
    print(f"Found {len(labels)} classes: {labels[:10]}{'...' if len(labels) > 10 else ''}")
    print(f"Total raw images: {len(samples)}")

    if samples:
        test_path, test_label = samples[0]
        test_img = cv2.imread(test_path, cv2.IMREAD_COLOR)
        print(f"Sample: label={test_label!r}  shape={None if test_img is None else test_img.shape}")

    # Step 2: verify (corrupted / blurry only — no duplicate check)
    qc_summary = verify_full_dataset(RAW_DIR, PROCESSED_DIR)
    qc_df = pd.DataFrame(qc_summary).T
    qc_df.index.name = "label"
    print(qc_df)

    # Step 3: label map
    label_to_idx, idx_to_label = build_label_map(PROCESSED_DIR, LABEL_MAP_PATH)
    label_to_idx, idx_to_label = load_label_map(LABEL_MAP_PATH)
    print(idx_to_label)

    # Step 4: resize/convert for the WHOLE dataset, saved to disk under CLEAN_DIR
    clean_samples, _ = build_clean_dataset(PROCESSED_DIR, CLEAN_DIR)

    # Save one preview figure (instead of a blocking popup) so you can
    # visually confirm the resizing worked
    if clean_samples:
        demo_path, demo_label = clean_samples[0]
        bgr = cv2.imread(str(PROCESSED_DIR / demo_label / Path(demo_path).name), cv2.IMREAD_COLOR)
        rgb = resize_and_convert(bgr)
        save_preview_grid(rgb, demo_label, PREVIEW_DIR / "resize_preview.png")

    # Step 5: illustrate tensor shape (sanity check only)
    img_hwc = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img_chw = np.transpose(img_hwc, (2, 0, 1))
    tensor = torch.from_numpy(img_chw).float() / 255.0
    fake_batch = torch.stack([tensor, tensor, tensor])
    print("single image tensor shape:", tensor.shape)
    print("batch tensor shape:       ", fake_batch.shape)

    # Step 6: generate + SAVE augmented images for the whole clean dataset
    augment_and_save_dataset(CLEAN_DIR, AUG_DIR, n_augments=N_AUGMENTS_PER_IMAGE)

    # Also save one augmentation preview grid for a quick visual check
    if clean_samples:
        demo_clean_path = str(CLEAN_DIR / demo_label / Path(demo_path).name)
        demo_rgb = cv2.cvtColor(cv2.imread(demo_clean_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        save_augmentation_preview(demo_rgb, demo_label, PREVIEW_DIR / "augmentation_preview.png")

    print("Done. No train/val/test split was performed — "
          "clean images are in 'data/clean', augmented images are in 'data/augmented'.")


if __name__ == "__main__":
    main()