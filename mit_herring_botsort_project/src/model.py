import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_detector(num_classes: int, model_name: str = 'fasterrcnn_resnet50_fpn', pretrained_backbone: bool = False):
    """Build a PyTorch detector.

    This project intentionally defaults to no pretrained COCO weights so students
    can understand true scratch training. For better research accuracy, set
    pretrained_backbone=True or initialize from COCO and fine-tune.
    """
    if model_name != 'fasterrcnn_resnet50_fpn':
        raise ValueError(f'Unsupported model: {model_name}')

    weights = None
    weights_backbone = None
    if pretrained_backbone:
        weights_backbone = torchvision.models.ResNet50_Weights.DEFAULT

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=5,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@torch.no_grad()
def predict(model, images, device, score_threshold=0.35):
    model.eval()
    images = [img.to(device) for img in images]
    outputs = model(images)
    cleaned = []
    for out in outputs:
        keep = out['scores'] >= score_threshold
        cleaned.append({k: v[keep].detach().cpu() for k, v in out.items()})
    return cleaned
