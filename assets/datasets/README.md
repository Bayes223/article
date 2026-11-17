# Datasets

The datasets are **not** shipped with this repository. Put them under this folder (or anywhere else
and point `path` in the dataset yaml to that location).

Expected layout (standard YOLO format):

```
datasets/
├── RUOD/
│   ├── images/
│   │   ├── train/*.jpg
│   │   └── valid/*.jpg
│   └── labels/
│       ├── train/*.txt
│       └── valid/*.txt
└── aquarium_pretrain/
    ├── train/{images,labels}/
    ├── valid/{images,labels}/
    └── test/{images,labels}/
```

Each label file holds one object per line: `class cx cy w h`, all coordinates normalised to [0, 1].

## RUOD (10 classes)

Real-world underwater scenes (colour cast, low illumination, reduced contrast, blurred detail).
Official split: 7,933 training / 467 validation / 933 test images, 10 classes:

```
corals, cuttlefish, diver, echinus, fish, holothurian, jellyfish, scallop, starfish, turtle
```

Place it as `datasets/RUOD/` and adjust `ruod.yaml`.

## Aquarium (7 classes)

Indoor tank scenes (glass refraction, specular highlights, non-uniform artificial light, dense
clusters). 448 / 127 / 63 images for train / val / test, 7 classes:

```
fish, jellyfish, penguin, puffin, shark, starfish, stingray
```

Place it as `datasets/aquarium_pretrain/` (each split has `images/` and `labels/`) and adjust
`aquarium.yaml`.

## Using an absolute path

`path` in a dataset yaml may be absolute, which avoids copying data into the repository:

```yaml
path: /data/underwater/aquarium_pretrain
train: train/images
val: valid/images
test: test/images
```
