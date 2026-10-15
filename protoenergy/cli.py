import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

from .method import ProtoEnergy
from .metrics import detection_metrics


class Images(Dataset):
    def __init__(self, x, y, training=False):
        self.x, self.y = x, y
        layers = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if training else []
        self.transform = transforms.Compose(layers + [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.x[index])), int(self.y[index])


def network():
    net = models.resnet18(weights=None, num_classes=10)
    net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


def prepare(args):
    source = datasets.CIFAR10(args.root, train=True, download=args.download)
    test = datasets.CIFAR10(args.root, train=False, download=args.download)
    y = np.array(source.targets)
    rng = np.random.default_rng(args.seed)
    calibration = np.concatenate([rng.permutation(np.flatnonzero(y == k))[:500] for k in range(10)])
    training = np.setdiff1d(np.arange(len(y)), calibration)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for split, index in [("train", training), ("cal", calibration)]:
        np.savez_compressed(out / f"{split}.npz", x=source.data[index], y=y[index])
    np.savez_compressed(out / "test.npz", x=test.data, y=test.targets)


def loader(root, split, args, training=False):
    data = np.load(Path(root) / f"{split}.npz", allow_pickle=False)
    return DataLoader(Images(data["x"], data["y"], training), batch_size=args.batch_size, shuffle=training)


def train(args):
    torch.manual_seed(args.seed)
    net = network().to(args.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history = []
    batches = loader(args.data, "train", args, True)
    for epoch in range(args.epochs):
        net.train()
        total, count = 0, 0
        for x, y in batches:
            x, y = x.to(args.device), y.to(args.device)
            loss = nn.functional.cross_entropy(net(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        schedule.step()
        history.append({"epoch": epoch + 1, "loss": total / count})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "classifier.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))


def extract(net, batches, device):
    embeddings, logits, labels = [], [], []
    encoder = nn.Sequential(*list(net.children())[:-1])
    with torch.no_grad():
        for x, y in batches:
            z = encoder(x.to(device)).flatten(1)
            embeddings.append(z.cpu().numpy())
            logits.append(net.fc(z).cpu().numpy())
            labels.append(y.numpy())
    return tuple(np.concatenate(v) for v in [embeddings, logits, labels])


def loaded(args):
    net = network().to(args.device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    return net.eval()


def calibrate(args):
    net = loaded(args)
    training, _, labels = extract(net, loader(args.data, "train", args), args.device)
    calibration, logits, _ = extract(net, loader(args.data, "cal", args), args.device)
    detector = ProtoEnergy(args.temperature, args.alpha).fit(training, labels, calibration, logits)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    detector.save(out / "detector.npz")
    np.savez_compressed(out / "training_features.npz", z=training, y=labels)


def ood_loader(row, args):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(3) if row["kind"] == "mnist" else transforms.Lambda(lambda x: x.convert("RGB")), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))])
    root, kind = row["root"], row["kind"]
    if kind == "cifar100":
        dataset = datasets.CIFAR100(root, train=False, transform=transform, download=args.download)
    elif kind == "mnist":
        dataset = datasets.MNIST(root, train=False, transform=transform, download=args.download)
    elif kind == "svhn":
        dataset = datasets.SVHN(root, split="test", transform=transform, download=args.download)
    elif kind == "folder":
        dataset = datasets.ImageFolder(root, transform=transform)
    else:
        raise ValueError(f"Unsupported OOD kind: {kind}")
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


def evaluate(args):
    net, detector = loaded(args), ProtoEnergy.load(args.detector)
    z, logits, labels = extract(net, loader(args.data, "test", args), args.device)
    begin = perf_counter()
    id_scores = detector.score(z, logits, args.mode)
    detector_ms = (perf_counter() - begin) * 1000 / len(z)
    threshold = detector.thresholds[args.mode]
    frr = {str(k): float(np.mean(id_scores[labels == k] > threshold)) for k in np.unique(labels)}
    test_point = np.quantile(id_scores, 0.95, method="higher")
    common_frr = {str(k): float(np.mean(id_scores[labels == k] > test_point)) for k in np.unique(labels)}
    rows = json.loads(Path(args.ood).read_text(encoding="utf-8"))
    results = {}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    distributions = {}
    for row in rows:
        oz, ol, _ = extract(net, ood_loader(row, args), args.device)
        score = detector.score(oz, ol, args.mode)
        results[row["name"]] = {"group": row["group"], **detection_metrics(id_scores, score, threshold, labels, logits.argmax(1))}
        distributions[row["name"]] = score
    macro = {}
    for group in ["near", "far", "overall"]:
        selected = [v for v in results.values() if v["group"] == group or group == "overall"]
        macro[group] = {key: float(np.mean([v[key] for v in selected])) if selected else None for key in ["auroc", "aupr_out", "oar_at_95id", "fpr95_ood_detection", "validation_locked_ood_acceptance"]}
    full = detector.score(z, logits, "full")
    no_radius = detector.score(z, logits, "no_radius")
    payload = {"datasets": results, "dataset_macro": macro, "id_accuracy": float(np.mean(logits.argmax(1) == labels)),
               "locked_class_frr": frr, "locked_frr_gap": max(frr.values()) - min(frr.values()),
               "common95_class_frr": common_frr, "common95_frr_gap": max(common_frr.values()) - min(common_frr.values()),
               "detector_ms_per_image": detector_ms, "validation_threshold": threshold,
               "predicted_class_calibration_support": {str(k): len(v[0]) for k, v in detector.references.items()},
               "radius_rank_max_difference": float(np.max(np.abs(full - no_radius))),
               "radius_rank_decision_disagreement": float(np.mean((full > detector.thresholds["full"]) != (no_radius > detector.thresholds["no_radius"])))}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    np.savez_compressed(out / "scores.npz", id_scores=id_scores, labels=labels, predicted=logits.argmax(1), **distributions)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(id_scores, bins=40, density=True, histtype="step", label="ID")
    for name, score in distributions.items():
        ax.hist(score, bins=40, density=True, histtype="step", label=name)
    ax.set(xlabel="OOD score", ylabel="Density")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "scores.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", default="data/raw")
    p.add_argument("--output", default="data/prepared")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download", action="store_true")
    p.set_defaults(function=prepare)
    for command in ["train", "calibrate", "evaluate"]:
        p = sub.add_parser(command)
        p.add_argument("--data", default="data/prepared")
        p.add_argument("--output", default=f"runs/{command}")
        p.add_argument("--batch-size", type=int, default=128)
        p.add_argument("--device", default="cpu")
        if command == "train":
            p.add_argument("--epochs", type=int, default=200)
            p.add_argument("--lr", type=float, default=0.1)
            p.add_argument("--seed", type=int, default=42)
            p.set_defaults(function=train)
        else:
            p.add_argument("--checkpoint", default="runs/train/classifier.pt")
            if command == "calibrate":
                p.add_argument("--temperature", type=float, default=1)
                p.add_argument("--alpha", type=float, default=0.5)
                p.set_defaults(function=calibrate)
            else:
                p.add_argument("--detector", default="runs/calibrate/detector.npz")
                p.add_argument("--ood", default="configs/ood.json")
                p.add_argument("--download", action="store_true")
                p.add_argument("--mode", choices=["full", "pooled", "no_radius", "energy", "prototype", "radius", "raw", "mean_radius", "quantile_radius", "msp", "maxlogit"], default="full")
                p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)
