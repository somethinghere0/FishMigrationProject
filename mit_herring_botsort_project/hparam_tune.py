from __future__ import annotations

"""Small Optuna example for hyperparameter tuning.

This intentionally runs short trials so students can learn the idea before launching long GPU jobs.
For real experiments, increase --epochs-per-trial and --trials.
"""

import argparse
from copy import deepcopy

import optuna
import torch
from torch.utils.data import DataLoader

from src.coco_dataset import CocoFishDataset, collate_fn, split_ids
from src.metrics import match_stats
from src.model import build_detector
from src.utils import load_config, resolve_device, seed_everything
from train import train_one_epoch, evaluate


def objective(trial, cfg, epochs_per_trial):
    local = deepcopy(cfg)
    local['train']['lr'] = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    local['train']['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    local['train']['batch_size'] = trial.suggest_categorical('batch_size', [1, 2, 4])
    local['inference']['score_threshold'] = trial.suggest_float('score_threshold', 0.2, 0.6)
    seed_everything(local.get('seed', 42))
    device = resolve_device(local.get('device', 'cuda'))

    train_ids, val_ids, _ = split_ids(local['paths']['coco_json'], local['train']['val_fraction'], local['train']['test_fraction'], local.get('seed', 42))
    # Use a small subset per trial for speed. Remove this slice for full studies.
    train_ids = train_ids[:800]
    val_ids = val_ids[:200]
    ds_kwargs = dict(coco_json=local['paths']['coco_json'], image_root=local['paths']['raw_root'],
                     class_mode=local['classes']['mode'], fish_category_names=local['classes']['fish_category_names'],
                     min_box_area=local['train']['min_box_area'])
    train_ds = CocoFishDataset(image_ids=train_ids, **ds_kwargs)
    val_ds = CocoFishDataset(image_ids=val_ids, **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=local['train']['batch_size'], shuffle=True, num_workers=local['train']['num_workers'], collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=local['train']['batch_size'], shuffle=False, num_workers=local['train']['num_workers'], collate_fn=collate_fn)
    model = build_detector(train_ds.num_classes, local['train']['model'], local['train']['pretrained_backbone']).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=local['train']['lr'], momentum=local['train']['momentum'], weight_decay=local['train']['weight_decay'])
    scaler = torch.cuda.amp.GradScaler() if local['train']['amp'] and device.type == 'cuda' else None
    for _ in range(epochs_per_trial):
        train_one_epoch(model, train_loader, optimizer, device, scaler, local['train']['gradient_clip_norm'])
    stats = evaluate(model, val_loader, device, local['inference']['score_threshold'])
    trial.set_user_attr('count_mae', stats['count_mae'])
    return stats['f1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--epochs-per-trial', type=int, default=3)
    args = parser.parse_args()
    cfg = load_config(args.config)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective(t, cfg, args.epochs_per_trial), n_trials=args.trials)
    print('Best F1:', study.best_value)
    print('Best params:', study.best_params)
    print('Best count_MAE:', study.best_trial.user_attrs.get('count_mae'))


if __name__ == '__main__':
    main()
