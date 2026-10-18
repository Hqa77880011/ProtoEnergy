# ProtoEnergy

Implementation of **ProtoEnergy: Class-Conditional Empirical Calibration of Energy and Prototype Geometry for Out-of-Distribution Detection**. A frozen classifier supplies energy and predicted-class feature deviation. ID-only empirical ranks place both signals on a common scale before fixed-weight fusion.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, torchvision 0.22.1, scikit-learn 1.6.1, matplotlib 3.10.0, pillow 12.3.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Download CIFAR-10 through TorchVision and reserve 500 ID calibration images per class using seed 42:

```bash
protoenergy prepare --root data/raw --download --output data/prepared --seed 42
```

`configs/ood.json` describes the six benchmark sources. CIFAR-100, MNIST and SVHN use TorchVision datasets. Obtain TinyImageNet, Describable Textures and Places365 from their maintainers and place their evaluation images under class subdirectories so `ImageFolder` can read them:

```text
data/ood/tinyimagenet/class_name/image.png
data/ood/textures/class_name/image.jpg
data/ood/places365/class_name/image.jpg
```

For TinyImageNet's released validation directory, organize images by their validation class annotations before using `ImageFolder`. Folder labels are ignored for OOD scoring. The manifest's `root` paths are relative to the current working directory; `group` is `near` or `far`. Edit the paths to match your downloads. All sources are converted to RGB and resized to 32×32 with the same classifier normalization.



## Training and inference

```bash
protoenergy train --data data/prepared --output runs/train --seed 42
protoenergy calibrate --data data/prepared --checkpoint runs/train/classifier.pt --output runs/calibrate
```

## Evaluation

```bash
protoenergy evaluate --data data/prepared --checkpoint runs/train/classifier.pt --detector runs/calibrate/detector.npz --ood configs/ood.json --download --output runs/full
```

`--download` obtains only TorchVision datasets; folder sources must exist at the configured paths.

## Comparisons and ablations

```bash
protoenergy evaluate --data data/prepared --checkpoint runs/train/classifier.pt --detector runs/calibrate/detector.npz --ood configs/ood.json --download --mode energy --output runs/energy
protoenergy evaluate --data data/prepared --checkpoint runs/train/classifier.pt --detector runs/calibrate/detector.npz --ood configs/ood.json --download --mode msp --output runs/msp
protoenergy evaluate --data data/prepared --checkpoint runs/train/classifier.pt --detector runs/calibrate/detector.npz --ood configs/ood.json --download --mode maxlogit --output runs/maxlogit
protoenergy evaluate --data data/prepared --checkpoint runs/train/classifier.pt --detector runs/calibrate/detector.npz --ood configs/ood.json --download --mode prototype --output runs/prototype
protoenergy evaluate --data data/prepared --checkpoint runs/train/classifier.pt --detector runs/calibrate/detector.npz --ood configs/ood.json --download --mode radius --output runs/radius
protoenergy evaluate --data data/prepared --checkpoint runs/train/classifier.pt --detector runs/calibrate/detector.npz --ood configs/ood.json --download --mode raw --output runs/raw
protoenergy evaluate --data data/prepared --checkpoint runs/train/classifier.pt --detector runs/calibrate/detector.npz --ood configs/ood.json --download --mode mean_radius --output runs/mean_radius
protoenergy evaluate --data data/prepared --checkpoint runs/train/classifier.pt --detector runs/calibrate/detector.npz --ood configs/ood.json --download --mode quantile_radius --output runs/quantile_radius
protoenergy evaluate --data data/prepared --checkpoint runs/train/classifier.pt --detector runs/calibrate/detector.npz --ood configs/ood.json --download --mode pooled --output runs/pooled
protoenergy evaluate --data data/prepared --checkpoint runs/train/classifier.pt --detector runs/calibrate/detector.npz --ood configs/ood.json --download --mode no_radius --output runs/no_radius
```

## Outputs

`runs/train/classifier.pt` and `history.json` retain fitting outputs. `runs/calibrate/detector.npz` and `training_features.npz` retain ID-only calibration. Evaluation writes `metrics.json`, `scores.npz` and `scores.png`.
