import argparse
from pathlib import Path
from collections import Counter


def validate(dataset_dir: Path, expected_class: str = "0"):
    labels = list(dataset_dir.glob("**/labels/*.txt"))
    counts = Counter()
    bad = []
    malformed = []
    for f in labels:
        for line_no, line in enumerate(f.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                malformed.append((str(f), line_no, line))
                continue
            cid = parts[0]
            counts[cid] += 1
            if cid != expected_class:
                bad.append((str(f), line_no, line))
    print("Dataset folder:", dataset_dir)
    print("Label files found:", len(labels))
    print("Class IDs found:", counts)
    print("Bad non-zero label lines:", bad[:30])
    print("Number of bad non-zero label lines:", len(bad))
    print("Malformed lines:", malformed[:30])
    print("Number of malformed lines:", len(malformed))
    if bad or malformed:
        raise SystemExit("Label validation failed. Fix labels before training.")
    print("OK: all non-empty YOLO labels use class 0 only.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/herring_yolo")
    args = ap.parse_args()
    validate(Path(args.dataset))
