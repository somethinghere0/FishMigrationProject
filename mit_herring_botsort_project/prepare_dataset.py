from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from src.preprocessing import preprocess_file
from src.utils import load_config, ensure_dir


def copy_or_preprocess_images(coco_json: Path, raw_root: Path, processed_root: Path, pre_cfg: dict, limit: int | None = None):
    with open(coco_json, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    images = coco['images'][:limit] if limit else coco['images']
    new_images = []
    for img in tqdm(images, desc='preprocess/copy images'):
        src = raw_root / img['file_name']
        if not src.exists():
            matches = list(raw_root.rglob(Path(img['file_name']).name))
            if not matches:
                print(f'Missing image, skipping: {src}')
                continue
            src = matches[0]
        dst = processed_root / img['file_name']
        if pre_cfg.get('enabled', True):
            preprocess_file(src, dst, pre_cfg)
            out = cv2.imread(str(dst))
            if out is not None:
                img = dict(img)
                img['height'], img['width'] = out.shape[:2]
                # Warning: if resizing is enabled, bounding boxes need scaling.
                # For precise training, keep resize_long_side null in preprocessing and resize in model transforms.
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        new_images.append(img)

    valid_ids = {img['id'] for img in new_images}
    coco['images'] = new_images
    coco['annotations'] = [ann for ann in coco['annotations'] if ann['image_id'] in valid_ids]
    out_json = processed_root / 'metadata_preprocessed.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(coco, f)
    print(f'Wrote {out_json}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--limit', type=int, default=None, help='Use a small limit for class demos/sanity checks.')
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_root = Path(cfg['paths']['raw_root'])
    coco_json = Path(cfg['paths']['coco_json'])
    processed_root = ensure_dir(cfg['paths']['processed_root'])
    copy_or_preprocess_images(coco_json, raw_root, processed_root, cfg['preprocessing'], args.limit)
    print('Important: if resize_long_side changed image sizes, scale COCO boxes before training. For teaching, prefer no offline resizing; resizing is safer inside the model pipeline.')


if __name__ == '__main__':
    main()
