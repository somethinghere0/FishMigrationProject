from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.coco_dataset import CocoFishDataset, collate_fn, split_ids
from src.metrics import match_stats, simple_count_mae
from src.model import build_detector
from src.utils import ensure_dir, load_config, resolve_device, seed_everything


def train_one_epoch(model, loader, optimizer, device, scaler=None, grad_clip_norm=5.0):
    model.train()
    total = 0.0
    for images, targets in tqdm(loader, desc='train', leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, score_threshold):
    model.eval()
    all_outputs, all_targets = [], []
    for images, targets in tqdm(loader, desc='val', leave=False):
        images_gpu = [img.to(device) for img in images]
        outputs = model(images_gpu)
        all_outputs.extend([{k: v.detach().cpu() for k, v in o.items()} for o in outputs])
        all_targets.extend(targets)
    stats = match_stats(all_outputs, all_targets, score_threshold=score_threshold)
    stats['count_mae'] = simple_count_mae(all_outputs, all_targets, score_threshold=score_threshold)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get('seed', 42))
    device = resolve_device(cfg.get('device', 'cuda'))

    train_ids, val_ids, test_ids = split_ids(
        cfg['paths']['coco_json'],
        cfg['train']['val_fraction'],
        cfg['train']['test_fraction'],
        cfg.get('seed', 42),
    )
    ds_kwargs = dict(
        coco_json=cfg['paths']['coco_json'],
        image_root=cfg['paths']['raw_root'],
        class_mode=cfg['classes']['mode'],
        fish_category_names=cfg['classes']['fish_category_names'],
        min_box_area=cfg['train']['min_box_area'],
    )
    train_ds = CocoFishDataset(image_ids=train_ids, **ds_kwargs)
    val_ds = CocoFishDataset(image_ids=val_ids, **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False,
                            num_workers=cfg['train']['num_workers'], collate_fn=collate_fn)

    model = build_detector(train_ds.num_classes, cfg['train']['model'], cfg['train']['pretrained_backbone']).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg['train']['lr'], momentum=cfg['train']['momentum'], weight_decay=cfg['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['train']['lr_step_size'], gamma=cfg['train']['lr_gamma'])
    scaler = torch.cuda.amp.GradScaler() if cfg['train']['amp'] and device.type == 'cuda' else None

    out_dir = ensure_dir(cfg['paths']['output_dir'])
    best_f1 = -1.0
    for epoch in range(1, cfg['train']['epochs'] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, scaler, cfg['train']['gradient_clip_norm'])
        scheduler.step()
        stats = evaluate(model, val_loader, device, cfg['inference']['score_threshold'])
        print(f"Epoch {epoch:03d} | loss={loss:.4f} | val_f1={stats['f1']:.3f} | P={stats['precision']:.3f} | R={stats['recall']:.3f} | count_MAE={stats['count_mae']:.2f}")
        ckpt = {'model': model.state_dict(), 'cfg': cfg, 'epoch': epoch, 'val_stats': stats}
        torch.save(ckpt, out_dir / 'last.pt')
        if stats['f1'] > best_f1:
            best_f1 = stats['f1']
            torch.save(ckpt, out_dir / 'best.pt')
    print(f'Done. Best checkpoint: {out_dir / "best.pt"}')


if __name__ == '__main__':
    main()
