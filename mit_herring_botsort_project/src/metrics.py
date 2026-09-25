from __future__ import annotations

import torch
from torchvision.ops import box_iou


def detection_counts_from_output(output, score_threshold=0.35):
    return int((output['scores'] >= score_threshold).sum().item())


def simple_count_mae(outputs, targets, score_threshold=0.35):
    abs_errors = []
    for out, tgt in zip(outputs, targets):
        pred_n = int((out['scores'].detach().cpu() >= score_threshold).sum().item())
        gt_n = int(tgt['boxes'].shape[0])
        abs_errors.append(abs(pred_n - gt_n))
    return sum(abs_errors) / max(len(abs_errors), 1)


@torch.no_grad()
def match_stats(outputs, targets, score_threshold=0.35, iou_threshold=0.5):
    tp = fp = fn = 0
    for out, tgt in zip(outputs, targets):
        keep = out['scores'].detach().cpu() >= score_threshold
        pred_boxes = out['boxes'].detach().cpu()[keep]
        gt_boxes = tgt['boxes'].detach().cpu()
        if len(pred_boxes) == 0:
            fn += len(gt_boxes)
            continue
        if len(gt_boxes) == 0:
            fp += len(pred_boxes)
            continue
        ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = set()
        for i in range(ious.shape[0]):
            j = int(torch.argmax(ious[i]).item())
            if float(ious[i, j]) >= iou_threshold and j not in matched_gt:
                tp += 1
                matched_gt.add(j)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}
