# -*- coding: utf-8 -*-
"""Training entry point for FPD (CSConv) + DHTA.

The detector is defined entirely by `ultralytics/cfg/models/v8/yolov8_fin.yaml`:
FPD replaces the strided stem, and three DHTA blocks are inserted on the P3/P4/P5
neck outputs (E = 32 hyperedges each).

Examples
--------
# RUOD, YOLOv8n-scale model, single GPU
python train.py --data ruod.yaml --epochs 100 --batch 32 --device 0

# Aquarium, initialised from a checkpoint of the same architecture
python train.py --data aquarium.yaml --epochs 600 --pretrained runs/train/fpd_dhta/weights/best.pt

# Resume an interrupted run
python train.py --data ruod.yaml --resume
"""

import argparse

from ultralytics import YOLO

DEFAULT_MODEL = "ultralytics/cfg/models/v8/yolov8_fin.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="Train FPD + DHTA on an underwater detection dataset")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model yaml (default: our FPD+DHTA config)")
    p.add_argument("--data", default="ruod.yaml", help="dataset yaml (ruod.yaml / aquarium.yaml)")
    p.add_argument("--epochs", type=int, default=100, help="training epochs (100 for RUOD, 600 for Aquarium)")
    p.add_argument("--batch", type=int, default=32, help="total batch size")
    p.add_argument("--imgsz", type=int, default=640, help="input resolution")
    p.add_argument("--device", default="0", help="CUDA device id, e.g. 0 or 0,1; 'cpu' for CPU")
    p.add_argument("--workers", type=int, default=8, help="dataloader workers")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--project", default="runs/train", help="output directory")
    p.add_argument("--name", default="fpd_dhta", help="run name")
    p.add_argument("--pretrained", default=None, help="optional checkpoint of the SAME architecture to initialise from")
    p.add_argument("--resume", action="store_true", help="resume the last run in --project/--name")
    return p.parse_args()


def main():
    a = parse_args()

    model = YOLO(a.model)
    if a.pretrained and not a.resume:
        model.load(a.pretrained)

    model.train(
        data=a.data,
        epochs=a.epochs,
        batch=a.batch,
        imgsz=a.imgsz,
        device=a.device,
        workers=a.workers,
        seed=a.seed,
        deterministic=True,
        project=a.project,
        name=a.name,
        exist_ok=True,
        resume=a.resume,
        # augmentation follows the Ultralytics defaults, which are the settings
        # used for all experiments in the paper:
        #   mosaic 1.0 (disabled for the last 10 epochs), scale 0.5, fliplr 0.5,
        #   hsv_h/s/v 0.015/0.7/0.4, mixup 0.0, erasing 0.4, randaugment
    )


if __name__ == "__main__":
    main()
