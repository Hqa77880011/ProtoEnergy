# ProtoEnergy

Implementation of **ProtoEnergy: Class-Conditional Empirical Calibration of Energy and Prototype Geometry for Out-of-Distribution Detection**. A frozen classifier supplies energy and predicted-class feature deviation. ID-only empirical ranks place both signals on a common scale before fixed-weight fusion.

## Install and train the classifier

Use Python 3.10+ and activate a virtual environment:

```bash
python -m venv .venv
python -m pip install -e .
protoenergy prepare --root data/raw --download --output data/prepared
protoenergy train --data data/prepared --output runs/train
protoenergy calibrate --data data/prepared --checkpoint runs/train/classifier.pt --output runs/calibrate
```

Preparation obtains CIFAR-10 through TorchVision and reserves exactly 500 images per class for detector calibration. The remaining 45,000 images train a CIFAR-style ResNet-18 with a 3×3, stride-one stem and no initial max pooling. The official test set remains untouched until evaluation. `--seed` fixes the stratified split; use a separate prepared directory and run directory for each seed.

Training uses cross-entropy, SGD with momentum 0.9, weight decay 0.0005 and cosine decay. `--epochs` defaults to 200, `--lr` to 0.1 and `--batch-size` to 128. The fixed final epoch is saved without selecting on detector calibration labels. `--device cuda` enables a compatible GPU installation.

Calibration reads only ID training features and held-out ID calibration inputs. It computes normalized class prototypes, 95th-percentile radii, predicted-class empirical CDFs and the fused-score 95th-percentile threshold. `--temperature` defaults to 1 and `--alpha` to the paper's fixed 0.5 fusion weight. Neither setting is selected on OOD data. Every predicted class must have ID calibration support; an empty group raises an explicit error.

## Prepare OOD sources

`configs/ood.json` describes the six benchmark sources. CIFAR-100, MNIST and SVHN use TorchVision datasets. Obtain TinyImageNet, Describable Textures and Places365 from their maintainers and place their evaluation images under class subdirectories so `ImageFolder` can read them:

```text
data/ood/tinyimagenet/class_name/image.png
data/ood/textures/class_name/image.jpg
data/ood/places365/class_name/image.jpg
```

For TinyImageNet's released validation directory, organize images by their validation class annotations before using `ImageFolder`. Folder labels are ignored for OOD scoring. The manifest's `root` paths are relative to the current working directory; `group` is `near` or `far`. Edit the paths to match your downloads. All sources are converted to RGB and resized to 32×32 with the same classifier normalization.

```bash
protoenergy evaluate --ood configs/ood.json --download --output runs/full
```

`--download` applies only to the TorchVision sources. It does not download or replace the folder datasets. All OOD inputs are used only for evaluation.

## Common-checkpoint comparisons

```bash
protoenergy evaluate --mode energy --output runs/energy
protoenergy evaluate --mode msp --output runs/msp
protoenergy evaluate --mode maxlogit --output runs/maxlogit
protoenergy evaluate --mode prototype --output runs/prototype
protoenergy evaluate --mode radius --output runs/radius
protoenergy evaluate --mode raw --output runs/raw
protoenergy evaluate --mode mean_radius --output runs/mean_radius
protoenergy evaluate --mode quantile_radius --output runs/quantile_radius
protoenergy evaluate --mode pooled --output runs/pooled
protoenergy evaluate --mode no_radius --output runs/no_radius
```

All modes load the same classifier and ID-only detector. Raw fusion combines energy with unscaled distance; mean-radius and quantile-radius modes change only the geometric unit. Full and no-radius class-conditional ranks are equivalent because a positive class-specific scale preserves ordering. The implementation ranks raw distances for both variants to avoid artificial floating-point tie changes. Pooled calibration instead uses a global ID reference and can change cross-class ordering.

## Outputs and operating points

Training writes `classifier.pt` and `history.json`. Calibration writes `detector.npz` and extracted training features. Evaluation writes `metrics.json`, per-dataset `scores.npz`, and `scores.png`.

The metric file reports AUROC, AUPR-OUT, three distinct operating points, ID accuracy, class-wise false rejection, calibration support, detector-only time and rank-invariance diagnostics:

- `fpr95_ood_detection`: ID rejection at a curve threshold achieving at least 95% OOD detection.
- `oar_at_95id`: OOD acceptance at the ID-test 95th-percentile curve threshold. `curve_id_acceptance` reports its achieved acceptance when score ties prevent exact 95%.
- `validation_locked_ood_acceptance`: OOD acceptance under the threshold selected from ID calibration and applied unchanged. Its companion `validation_locked_id_frr` is evaluated on ID test inputs.

Near, far and overall summaries average dataset metrics equally, with two near and four far datasets in the default manifest. They do not pool OOD images across datasets. `locked_class_frr` and its gap use true ID classes under the deployment threshold. `common95_class_frr` uses the separate ID-test curve threshold. All ratios are fractions. `oscr` additionally combines correct known classification with unknown acceptance.

## Implementation details

Prototype geometry is conditioned on the classifier's predicted class at calibration and inference; training labels define prototypes and radii. ECDF ties use the right-continuous `<=` convention. Score thresholds use NumPy's empirical 95th percentile. The final score is an extremeness rank, not a calibrated probability of being OOD. Detector timing excludes classifier extraction and is reported per image for the cached test feature batch. The provided baseline modes cover output confidence and prototype/calibration controls; external baseline names are not assigned to approximations.
