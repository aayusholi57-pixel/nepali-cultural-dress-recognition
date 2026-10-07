from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.hybrid_model import NepaliDressHybridClassifier


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DATA_DIR = Path("data")
MODEL_DIR = Path("models/hybrid")
MODEL_PATH = MODEL_DIR / "best_hybrid_v1.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.8, 1.0),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        DATA_DIR / "val",
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def make_class_weights(dataset):
    targets = np.asarray(dataset.targets)
    counts = np.bincount(
        targets,
        minlength=len(dataset.classes),
    )

    # Inverse-square-root weighting is less aggressive than
    # pure inverse-frequency weighting for small classes.
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return accuracy, macro_f1


def main():
    set_seed(SEED)

    torch.set_num_threads(4)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = build_dataloaders()

    if train_dataset.classes != val_dataset.classes:
        raise RuntimeError(
            "Train and validation class mappings do not match."
        )

    num_classes = len(train_dataset.classes)

    print(f"Classes: {num_classes}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print("\nClasses:")
    for index, name in enumerate(train_dataset.classes):
        print(f"  {index} -> {name}")

    model = NepaliDressHybridClassifier(
        num_classes=num_classes,
        dropout_rate=0.4,
        pretrained=True,
    ).to(device)

    print(
        "\nSelected ResNet feature shape:",
        model.feature_shape(IMAGE_SIZE),
    )

    trainable = [
        param for param in model.parameters()
        if param.requires_grad
    ]
    frozen = [
        param for param in model.parameters()
        if not param.requires_grad
    ]

    print(
        f"Trainable parameters: "
        f"{sum(p.numel() for p in trainable):,}"
    )
    print(
        f"Frozen parameters: "
        f"{sum(p.numel() for p in frozen):,}"
    )

    class_weights = make_class_weights(
        train_dataset
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_macro_f1 = -1.0
    best_epoch = -1

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        sample_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        train_loss = running_loss / sample_count
        val_accuracy, val_macro_f1 = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch

            checkpoint = {
                "artifact_version": "hybrid_v1",
                "model_name": "resnet50_selected_hybrid",
                "architecture": {
                    "backbone": "ResNet50 pretrained",
                    "selected_backbone": [
                        "conv1",
                        "bn1",
                        "relu",
                        "maxpool",
                        "layer1",
                        "layer2",
                    ],
                    "excluded_backbone": [
                        "layer3",
                        "layer4",
                        "avgpool",
                        "fc",
                    ],
                    "custom_cnn": [
                        "Conv2d 512->256",
                        "BatchNorm2d 256",
                        "ReLU",
                        "Conv2d 256->128",
                        "BatchNorm2d 128",
                        "ReLU",
                    ],
                    "classifier": [
                        "AdaptiveAvgPool2d",
                        "Linear 128->128",
                        "ReLU",
                        "Dropout 0.4",
                        f"Linear 128->{num_classes}",
                    ],
                },
                "state_dict": model.state_dict(),
                "class_names": train_dataset.classes,
                "num_classes": num_classes,
                "image_size": IMAGE_SIZE,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "epoch": epoch,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
                "backbone_frozen": True,
            }

            torch.save(checkpoint, MODEL_PATH)

            print(
                f"  -> Saved best model: {MODEL_PATH}"
            )

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Macro F1: {best_macro_f1:.4f}")
    print(f"Model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
