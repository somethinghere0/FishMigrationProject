from __future__ import annotations
import argparse
from src.fish_counter.utils import load_yaml
from src.fish_counter.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Pretrained fish detector -> ByteTrack-style tracker -> herring count")
    parser.add_argument("--config", default="configs/cfd_counter.yaml", help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
