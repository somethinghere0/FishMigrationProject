import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare(images_dir: Path, labels_dir: Path, out_dir: Path, val_ratio: float, test_ratio: float, seed: int):
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test_imgs = images[:n_test]
    val_imgs = images[n_test:n_test+n_val]
    train_imgs = images[n_test+n_val:]

    splits = {"train": train_imgs, "valid": val_imgs, "test": test_imgs}
    for split, imgs in splits.items():
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, out_dir / split / "images" / img.name)
            label = labels_dir / f"{img.stem}.txt"
            dest_label = out_dir / split / "labels" / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, dest_label)
            else:
                # Empty frame: create empty label file. This is important for false-positive reduction.
                dest_label.write_text("")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "path: .\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names:\n"
        "  0: fish\n"
    )
    print(f"Prepared YOLO dataset at: {out_dir}")
    for split, imgs in splits.items():
        print(f"  {split}: {len(imgs)} images")
    print("\nNext: run scripts/train_cfd_rfdetr.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="frames_labeled/images")
    ap.add_argument("--labels", default="frames_labeled/labels")
    ap.add_argument("--out", default="datasets/herring_yolo")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    prepare(Path(args.images), Path(args.labels), Path(args.out), args.val_ratio, args.test_ratio, args.seed)
