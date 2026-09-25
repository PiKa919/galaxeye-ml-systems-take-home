"""Train a seven-class head on a pretrained MobileNetV3-Small backbone."""

import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from galaxeye.contracts import CLASSES
from galaxeye.data import ROOT, SEED, SPLITS

OUTPUT = ROOT / "artifacts" / "v2"
SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def transform(training: bool) -> transforms.Compose:
    operations = [transforms.Resize((SIZE, SIZE))]
    if training:
        operations += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    operations += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(operations)


def make_model(pretrained: bool) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    return model


def train() -> Path:
    split = json.loads((SPLITS / "manifest.json").read_text())
    torch.manual_seed(SEED)
    random.seed(SEED)
    training = datasets.ImageFolder(SPLITS / "train", transform=transform(True))
    validation = datasets.ImageFolder(SPLITS / "val", transform=transform(False))
    if tuple(training.classes) != CLASSES or tuple(validation.classes) != CLASSES:
        raise ValueError("Dataset class order differs from the version contract")
    train_loader = DataLoader(training, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(validation, batch_size=32, shuffle=False, num_workers=0)
    model = make_model(pretrained=True)
    model.features.requires_grad_(False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=0.001, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "model.pt"
    best_accuracy = -1.0
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, 16):
        model.train()
        model.features.eval()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(labels)
        model.eval()
        correct = 0
        count = 0
        val_loss = 0.0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += float(criterion(logits, labels).cpu()) * len(labels)
                correct += int((logits.argmax(dim=1) == labels).sum().cpu())
                count += len(labels)
        accuracy = correct / count
        val_loss /= count
        print(
            f"v2 epoch {epoch:02d}: train_loss={total_loss / len(training):.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={accuracy:.4f}",
            flush=True,
        )
        if accuracy > best_accuracy or (
            accuracy == best_accuracy and val_loss < best_loss
        ):
            best_accuracy, best_loss, stale_epochs = accuracy, val_loss, 0
            torch.save(model.cpu().state_dict(), target)
            model.to(device)
        else:
            stale_epochs += 1
            if stale_epochs >= 4:
                break

    manifest = {
        "version": "v2",
        "architecture": "MobileNetV3-Small/ImageNet-head-only",
        "weights_file": target.name,
        "weights_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "classes": list(CLASSES),
        "split_digest": split["digest"],
        "image_size": SIZE,
        "preprocess_id": "resize-full-tile-224-imagenet-normalize",
        "mean": list(MEAN),
        "std": list(STD),
        "best_val_accuracy": best_accuracy,
        "review_score_threshold": 0.6,
        "review_threshold_status": "provisional; score is not calibrated confidence",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target
