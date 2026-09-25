import argparse
from pathlib import Path
import yaml


def build_model(model_size: str, pretrain_weights: str):
    # RF-DETR package names may evolve; this keeps error messages student-friendly.
    try:
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
    except Exception as exc:
        raise ImportError(
            "Could not import RF-DETR. Install dependencies with: pip install -r requirements.txt\n"
            f"Original error: {exc}"
        )

    model_size = model_size.lower().strip()
    cls_map = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
    }
    if model_size not in cls_map:
        raise ValueError("model_size must be one of: nano, small, medium")

    if pretrain_weights and Path(pretrain_weights).exists():
        print(f"Loading CFD pretrained weights: {pretrain_weights}")
        return cls_map[model_size](pretrain_weights=str(pretrain_weights))
    else:
        raise FileNotFoundError(
            f"CFD pretrained checkpoint not found: {pretrain_weights}\n"
            "Download the matching CFD RF-DETR checkpoint from the Community Fish Detector Releases page "
            "and place it in the weights/ folder."
        )


def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    dataset_dir = Path(cfg["dataset_dir"])
    if not (dataset_dir / "data.yaml").exists():
        raise FileNotFoundError(
            f"YOLO dataset not found at {dataset_dir}. Expected data.yaml.\n"
            "Run scripts/prepare_yolo_dataset.py first."
        )

    model = build_model(cfg.get("model_size", "nano"), cfg["pretrain_weights"])

    print("\nStarting CFD RF-DETR fine-tuning")
    print("--------------------------------")
    print(f"Dataset: {dataset_dir}")
    print(f"Output:  {cfg['output_dir']}")
    print(f"Epochs:  {cfg.get('epochs', 25)}")
    print(f"Batch:   {cfg.get('batch_size', 2)}")
    print(f"Grad accumulation: {cfg.get('grad_accum_steps', 8)}")
    print(f"LR:      {cfg.get('lr', 5e-5)}")
    print(f"Encoder LR: {cfg.get('lr_encoder', 1e-5)}")

    # RF-DETR automatically detects YOLO format via data.yaml + train/images folders.
    # We use a smaller LR for encoder/backbone to make this behave like conservative fine-tuning.
    train_kwargs = dict(
        dataset_dir=str(dataset_dir),
        epochs=int(cfg.get("epochs", 25)),
        batch_size=int(cfg.get("batch_size", 2)),
        grad_accum_steps=int(cfg.get("grad_accum_steps", 8)),
        lr=float(cfg.get("lr", 5e-5)),
        output_dir=str(cfg.get("output_dir", "outputs/cfd_herring_finetune")),
    )
    # Most RF-DETR versions accept lr_encoder; if not, retry without it.
    lr_encoder = cfg.get("lr_encoder", None)
    if lr_encoder is not None:
        train_kwargs["lr_encoder"] = float(lr_encoder)

    try:
        model.train(**train_kwargs)
    except TypeError as exc:
        if "lr_encoder" in train_kwargs:
            print("This installed RF-DETR version did not accept lr_encoder; retrying without it.")
            train_kwargs.pop("lr_encoder", None)
            model.train(**train_kwargs)
        else:
            raise exc

    print("\nTraining complete.")
    print("Look for checkpoints such as checkpoint_best_total.pth inside the output directory.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    args = ap.parse_args()
    main(args.config)
