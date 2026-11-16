# FPD + DHTA: Frequency-Preserving Downsampling and Dynamic Hypergraph Token Attention for Underwater Object Detection

This repository contains the reference implementation of the two plug-and-play modules proposed for
lightweight underwater object detection:

| Module | File | What it does |
|---|---|---|
| **FPD** — Frequency-Preserving Downsampling | `ultralytics/nn/modules/wave.py` (`CSConv`) | Replaces the strided stem convolution with a **parameter-free Haar wavelet decomposition**. The input is split into four sub-bands (LL/HL/LH/HH) and re-organised, so structural detail is retained while the low-frequency degradation (scattering, light absorption) is separated from the high-frequency discriminative cues. |
| **DHTA** — Dynamic Hypergraph Token Attention | `ultralytics/nn/modules/dhta.py` (`DHTA`) | Builds **E = 32 hyperedges** from the token set with an input-adaptive incidence matrix, aggregates tokens into hyperedges, refines them with a hyperedge MLP, and broadcasts them back to every node. Cost is linear in the number of tokens, so it is affordable at all three detection scales. |

Both modules are integrated into a single YOLOv8n-scale detector defined by
`ultralytics/cfg/models/v8/yolov8_fin.yaml`; nothing else needs to be changed to reproduce the model.

---

## 1. Repository structure

```
yolov13/
├── train.py                                  # the only training entry point (this work)
├── ultralytics/
│   ├── cfg/models/v8/
│   │   ├── yolov8_fin.yaml                   # our detector: FPD stem + 3× DHTA (P3/P4/P5)
│   │   └── yolov8.yaml                       # plain YOLOv8 architecture, for reference
│   ├── nn/modules/
│   │   ├── wave.py                           # FPD / CSConv (Haar downsampling)
│   │   └── dhta.py                           # DHTA
│   └── ...                                   # Ultralytics engine (trainer, validator, ops)
├── datasets/README.md                        # expected data layout (datasets are not shipped)
├── ruod.yaml                                 # RUOD dataset config (10 classes)
├── aquarium.yaml                             # Aquarium dataset config (7 classes)
├── requirements.txt / pyproject.toml
└── LICENSE
```

## 2. Architecture

```
input 640×640
└── CSConv (FPD: Haar DWT, stride 2)          P1/2
    └── Conv ×4 + C2f (YOLOv8 backbone)      P2/4 … P5/32
        └── SPPF
            └── PAN neck (upsample + concat)
                ├── C2f → DHTA(E=32)  → P3  80×80×64
                ├── C2f → DHTA(E=32)  → P4  40×40×128
                └── C2f → DHTA(E=32)  → P5  20×20×256
                                             └── Detect
```

Per-module cost (measured by forward hooks, 640×640 input):

| Stage | H×W | C | E | hidden dim d | module params |
|---|---|---|---|---|---|
| P3 | 80×80 | 64 | 32 | 64 | 22 784 |
| P4 | 40×40 | 128 | 32 | 128 | 86 528 |
| P5 | 20×20 | 256 | 32 | 256 | 336 896 |

Total DHTA: **0.446 M parameters** (3 modules, 12.9 % of the detector); FPD adds only **0.001 M**.
Whole model: **3.455 M parameters, 4.454 G MACs** (640×640, batch 1, FP32).

## 3. Installation

```bash
conda create -n fpd_dhta python=3.10 -y
conda activate fpd_dhta
pip install torch torchvision          # CUDA build matching your driver
pip install -e .                       # installs the vendored ultralytics package + dependencies
```

Verified with PyTorch 2.4.1 + CUDA 12.1, Ultralytics 8.3.x, Python 3.10, RTX 3090.

## 4. Data

Labels follow the standard YOLO format (`class cx cy w h`, normalised). Point `path` in the dataset
yaml to your local copy:

```yaml
# ruod.yaml
path: datasets/RUOD        # <-- your path
train: images/train
val: images/valid
nc: 10
names: ['corals', 'cuttlefish', 'diver', 'echinus', 'fish',
        'holothurian', 'jellyfish', 'scallop', 'starfish', 'turtle']
```

`aquarium.yaml` expects the data at `datasets/aquarium_pretrain` (448 / 127 / 63 images for
train / val / test, 7 classes). The datasets themselves are **not** committed to this repository;
see `datasets/README.md` for the expected layout, or point `path` to an absolute location.

## 5. Training

Single entry point — `train.py`:

```bash
# RUOD (100 epochs, 1 GPU)
python train.py --data ruod.yaml --epochs 100 --batch 32 --device 0

# Aquarium (600 epochs)
python train.py --data aquarium.yaml --epochs 600 --batch 32 --device 0 --name fpd_dhta_aq

# Resume
python train.py --data ruod.yaml --resume
```

| Argument | Default | Meaning |
|---|---|---|
| `--model` | `ultralytics/cfg/models/v8/yolov8_fin.yaml` | our FPD + DHTA configuration |
| `--data` | `ruod.yaml` | dataset yaml |
| `--epochs` | 100 | 100 (RUOD) / 600 (Aquarium) in the paper |
| `--batch` / `--imgsz` | 32 / 640 | training batch and input resolution |
| `--device` | `0` | CUDA device(s) or `cpu` |
| `--seed` | 0 | seed (`deterministic=True` is enforced) |
| `--pretrained` | none | checkpoint of the **same** architecture to initialise from |
| `--project` / `--name` | `runs/train` / `fpd_dhta` | output location |

Input handling and augmentation are the Ultralytics defaults, which are the settings used in the
paper: aspect-preserving letterbox to 640×640 with grey-114 padding, pixel values scaled to [0,1],
mosaic 1.0 (disabled for the final 10 epochs), HSV jitter 0.015/0.7/0.4, horizontal flip 0.5,
affine scale 0.5, random erasing 0.4 and RandAugment; mixup is disabled.

## 6. Results

Main detector (YOLOv8n scale) with both modules, single run, same protocol for all rows:

| Dataset | Metric | Baseline | **+ FPD + DHTA** |
|---|---|---|---|
| RUOD (10 cls, 1 400-image val) | mAP50:95 (%) | 54.6 | **57.1** |
| Aquarium (7 cls, 63-image test) | mAP50:95 (%) | 37.39 | **38.89** |
| Cost | Params (M) | 3.008 | 3.455 |
| Cost | G MACs | 4.047 | 4.454 |
| Cost | Latency @640, RTX 3090 | 3.77 ms (265 FPS) | 5.71 ms (175 FPS) |

The gains come mainly from recall (missed detections 1 491 → 1 331, −10.7 % on RUOD) and
localisation; false positives are unchanged and class confusion rises slightly (75 → 82), which is
discussed in the paper as the main remaining limitation.

## 7. Notes on this release

* Only the proposed method is kept: comparison detectors (Hyper-YOLO, CEH-YOLO, AMSP, SeaNet,
  ablation variants) and their configs, training logs, checkpoints and analysis outputs have been
  removed from this copy to keep the repository compact and readable.
* The remaining code under `ultralytics/` is the training/validation engine the model is built on.

## 8. License

Released under the **AGPL-3.0** license inherited from the Ultralytics engine (see `LICENSE`).
If you use this code, please cite the paper.
