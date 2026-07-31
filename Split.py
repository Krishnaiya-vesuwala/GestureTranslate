import os
import shutil
import random
from pathlib import Path

SOURCE_DIR = "data/augmented"     # folder containing class subfolders (already cleaned images)
OUTPUT_DIR = "dataset"  

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
COPY_FILES = True  

random.seed(SEED)
def split_dataset():
    source_path = Path(SOURCE_DIR)
    output_path = Path(OUTPUT_DIR)

    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1.0"

    class_folders = sorted([d for d in source_path.iterdir() if d.is_dir()])
    if not class_folders:
        print(f"No class folders found in '{SOURCE_DIR}'. Check the path.")
        return

    summary = {}
    total_train, total_val, total_test = 0, 0, 0

    for class_folder in class_folders:
        class_name = class_folder.name

        # 1. Read images in this class
        images = [f for f in class_folder.iterdir()
                  if f.suffix.lower() in (".jpg", ".jpeg", ".png")]

        # 2. Count
        total_images = len(images)

        # 3. Shuffle (so split isn't biased by filename/capture order)
        random.shuffle(images)

        # 4. Split
        train_end = int(total_images * TRAIN_RATIO)
        val_end = train_end + int(total_images * VAL_RATIO)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        # 5. Create output folders
        train_dir = output_path / "train" / class_name
        val_dir = output_path / "validation" / class_name
        test_dir = output_path / "test" / class_name
        for d in (train_dir, val_dir, test_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 6. Copy (or move) images
        transfer_fn = shutil.copy2 if COPY_FILES else shutil.move
        for img in train_images:
            transfer_fn(str(img), str(train_dir / img.name))
        for img in val_images:
            transfer_fn(str(img), str(val_dir / img.name))
        for img in test_images:
            transfer_fn(str(img), str(test_dir / img.name))

        summary[class_name] = {
            "total": total_images,
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
        }
        total_train += len(train_images)
        total_val += len(val_images)
        total_test += len(test_images)

    # 7. Print summary
    print(f"{'Class':<15}{'Total':>8}{'Train':>8}{'Val':>8}{'Test':>8}")
    print("-" * 47)
    for class_name, counts in summary.items():
        print(f"{class_name:<15}{counts['total']:>8}{counts['train']:>8}{counts['val']:>8}{counts['test']:>8}")
    print("-" * 47)
    print(f"{'TOTAL':<15}{total_train + total_val + total_test:>8}{total_train:>8}{total_val:>8}{total_test:>8}")
    print(f"\nDone. Files were {'copied' if COPY_FILES else 'moved'} into '{OUTPUT_DIR}/train', "
          f"'{OUTPUT_DIR}/validation', '{OUTPUT_DIR}/test'.")


if __name__ == "__main__":
    split_dataset()



