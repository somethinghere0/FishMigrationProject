from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class CocoFishDataset(Dataset):
    def __init__(
        self,
        coco_json: str,
        image_root: str,
        image_ids: List[int] | None = None,
        class_mode: str = 'all_fish',
        fish_category_names: List[str] | None = None,
        min_box_area: float = 12,
        transforms=None,
    ):
        self.coco = COCO(coco_json)
        self.image_root = Path(image_root)
        self.class_mode = class_mode
        self.min_box_area = min_box_area
        self.transforms = transforms
        self.fish_category_names = [x.lower() for x in (fish_category_names or [])]

        all_ids = sorted(self.coco.imgs.keys()) if image_ids is None else list(image_ids)
        self.image_ids = all_ids

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_name = {c['id']: c['name'].lower() for c in cats}
        if class_mode == 'all_fish':
            self.cat_id_to_label = {cid: 1 for cid in self.cat_id_to_name}
            self.num_classes = 2  # background + fish
        else:
            sorted_cat_ids = sorted(self.cat_id_to_name.keys())
            self.cat_id_to_label = {cid: i + 1 for i, cid in enumerate(sorted_cat_ids)}
            self.num_classes = len(sorted_cat_ids) + 1

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        info = self.coco.imgs[img_id]
        file_name = info['file_name']
        img_path = self.image_root / file_name
        if not img_path.exists():
            # LILA metadata uses paths like Coonamessett/.../frame.PNG.
            # Some users put images one folder deeper; this fallback helps.
            candidates = list(self.image_root.rglob(Path(file_name).name))
            if not candidates:
                raise FileNotFoundError(f'Image not found: {img_path}')
            img_path = candidates[0]

        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            if bw * bh < self.min_box_area:
                continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            cat_id = ann['category_id']
            label = self.cat_id_to_label.get(cat_id, 1)
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            areas.append((x2 - x1) * (y2 - y1))
            iscrowd.append(int(ann.get('iscrowd', 0)))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        target = {
            'boxes': boxes.reshape(-1, 4),
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd,
        }
        image = F.to_tensor(image)
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def split_ids(coco_json: str, val_fraction: float = 0.15, test_fraction: float = 0.10, seed: int = 42):
    rng = np.random.default_rng(seed)
    coco = COCO(coco_json)
    ids = np.array(sorted(coco.imgs.keys()))
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_ids = ids[:n_test].tolist()
    val_ids = ids[n_test:n_test + n_val].tolist()
    train_ids = ids[n_test + n_val:].tolist()
    return train_ids, val_ids, test_ids
