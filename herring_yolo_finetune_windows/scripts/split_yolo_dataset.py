import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_dataset(source_images: Path, source_labels: Path, dataset_dir: Path, val_ratio: float, seed: int):
    if not source_images.exists():
        raise FileNotFoundError(f"Image folder not found: {source_images}")
    if not source_labels.exists():
        raise FileNotFoundError(f"Label folder not found: {source_labels}")

    images = sorted([p for p in source_images.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        raise RuntimeError(f"No images found in {source_images}")

    pairs = []
    missing = []
    for img in images:
        label = source_labels / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))
        else:
            # Empty-frame training is allowed in YOLO if you create an empty .txt.
            missing.append(img.name)
            empty_label = source_labels / f"{img.stem}.txt"
            empty_label.write_text("")
            pairs.append((img, empty_label))

    random.seed(seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 1 else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def copy_pairs(items, split):
        for img, label in items:
            shutil.copy2(img, dataset_dir / "images" / split / img.name)
            shutil.copy2(label, dataset_dir / "labels" / split / label.name)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")

    print(f"Total images: {len(pairs)}")
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    if missing:
        print(f"Created empty label files for {len(missing)} image(s) with no visible fish.")
    print(f"Dataset written to: {dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split labeled frames into YOLO train/val folders.")
    parser.add_argument("--source-images", default="frames_labeled/images", help="Folder containing labeled images")
    parser.add_argument("--source-labels", default="frames_labeled/labels", help="Folder containing YOLO .txt labels")
    parser.add_argument("--dataset-dir", default="dataset", help="Output YOLO dataset folder")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dataset(Path(args.source_images), Path(args.source_labels), Path(args.dataset_dir), args.val_ratio, args.seed)
