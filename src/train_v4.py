import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.v4_model import ResNet50CustomV4


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

MODEL_DIR = Path("models")
V4_MODEL_DIR = MODEL_DIR / "resnet50_custom_v4"

BEST_MODEL_PATH = V4_MODEL_DIR / "best_resnet50_custom_v4.pth"
HISTORY_PATH = V4_MODEL_DIR / "training_history_v4.json"

IMAGE_SIZE = 224
BATCH_SIZE = 8

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 5

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_LAYER4 = 1e-5

WEIGHT_DECAY = 1e-4

DROPOUT_RATE = 0.2

SEED = 42
CPU_THREADS = 4


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

torch.set_num_threads(CPU_THREADS)

print("=" * 70)
print("NEPALI CULTURAL DRESS RECOGNITION - V4 TRAINING")
print("=" * 70)
print()
print(f"Device: {device}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Phase 1 epochs: {PHASE1_EPOCHS}")
print(f"Phase 2 epochs: {PHASE2_EPOCHS}")
print()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose(
    [
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
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


# ============================================================
# DATASET
# ============================================================

print("Loading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform,
)

class_names = train_dataset.classes
num_classes = len(class_names)

print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Number of classes: {num_classes}")
print()

if train_dataset.classes != val_dataset.classes:
    raise RuntimeError(
        "Train and validation class lists do not match."
    )


print("Classes:")
for index, class_name in enumerate(class_names):
    print(f"  {index:2d} -> {class_name}")

print()


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("Calculating class weights...")

class_counts = np.bincount(
    train_dataset.targets,
    minlength=num_classes,
)

# Inverse square-root weighting reduces the effect of
# severe class imbalance without giving rare classes
# excessively large weights.
class_weights = 1.0 / np.sqrt(class_counts)

class_weights = class_weights / class_weights.mean()

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device,
)

print("Class counts:")
for index, count in enumerate(class_counts):
    print(f"  {class_names[index]:30s}: {count}")

print()


# ============================================================
# MODEL
# ============================================================

print("Creating V4 model...")

model = ResNet50CustomV4(
    num_classes=num_classes,
    dropout_rate=DROPOUT_RATE,
    pretrained=True,
)

model = model.to(device)

print("Model created successfully.")
print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return accuracy, macro_f1, weighted_f1


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, optimizer):
    model.train()

    running_loss = 0.0
    all_predictions = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            targets,
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_targets.extend(
            targets.detach().cpu().numpy()
        )

    epoch_loss = running_loss / len(loader.dataset)

    accuracy, macro_f1, weighted_f1 = calculate_metrics(
        all_targets,
        all_predictions,
    )

    return epoch_loss, accuracy, macro_f1, weighted_f1


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader):
    model.eval()

    running_loss = 0.0
    all_predictions = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            targets,
        )

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            targets.cpu().numpy()
        )

    epoch_loss = running_loss / len(loader.dataset)

    accuracy, macro_f1, weighted_f1 = calculate_metrics(
        all_targets,
        all_predictions,
    )

    return epoch_loss, accuracy, macro_f1, weighted_f1


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    epoch,
    phase,
    val_accuracy,
    val_macro_f1,
    val_weighted_f1,
):
    V4_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "artifact_version": "4.0",
        "model_name": "resnet50_custom_v4",
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes,
        "image_size": IMAGE_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "epoch": epoch,
        "phase": phase,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "val_weighted_f1": val_weighted_f1,
        "architecture": {
            "backbone": "pretrained_resnet50",
            "custom_conv1": "2048->512",
            "custom_conv2": "512->256",
            "global_average_pooling": True,
            "custom_linear1": "256->128",
            "dropout": DROPOUT_RATE,
            "custom_linear2": f"128->{num_classes}",
        },
    }

    torch.save(
        checkpoint,
        BEST_MODEL_PATH,
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "config": {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "phase1_epochs": PHASE1_EPOCHS,
        "phase2_epochs": PHASE2_EPOCHS,
        "learning_rate_head": LEARNING_RATE_HEAD,
        "learning_rate_layer4": LEARNING_RATE_LAYER4,
        "weight_decay": WEIGHT_DECAY,
        "dropout_rate": DROPOUT_RATE,
        "seed": SEED,
        "device": str(device),
    },
    "classes": class_names,
    "phase1": [],
    "phase2": [],
}


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_macro_f1 = -1.0
best_epoch = None
best_phase = None


# ============================================================
# PHASE 1
# ============================================================

print("=" * 70)
print("PHASE 1 - TRAIN CUSTOM HEAD")
print("=" * 70)
print()

model.freeze_backbone()

optimizer = torch.optim.AdamW(
    model.trainable_parameters(),
    lr=LEARNING_RATE_HEAD,
    weight_decay=WEIGHT_DECAY,
)

print("ResNet50 backbone: FROZEN")
print("Custom Conv + Linear layers: TRAINABLE")
print()

for epoch in range(1, PHASE1_EPOCHS + 1):

    train_loss, train_acc, train_macro_f1, train_weighted_f1 = (
        train_one_epoch(
            model,
            train_loader,
            optimizer,
        )
    )

    val_loss, val_acc, val_macro_f1, val_weighted_f1 = (
        validate(
            model,
            val_loader,
        )
    )

    epoch_data = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_macro_f1": train_macro_f1,
        "train_weighted_f1": train_weighted_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1,
        "val_weighted_f1": val_weighted_f1,
    }

    history["phase1"].append(epoch_data)

    print(
        f"Phase 1 | Epoch "
        f"{epoch:02d}/{PHASE1_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Train Macro F1: {train_macro_f1:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Macro F1: {val_macro_f1:.4f} | "
        f"Val Weighted F1: {val_weighted_f1:.4f}"
    )

    if val_macro_f1 > best_macro_f1:

        best_macro_f1 = val_macro_f1
        best_epoch = epoch
        best_phase = 1

        save_checkpoint(
            model,
            epoch,
            phase=1,
            val_accuracy=val_acc,
            val_macro_f1=val_macro_f1,
            val_weighted_f1=val_weighted_f1,
        )

        print(
            f"  ✓ New best model saved "
            f"(Macro F1: {val_macro_f1:.4f})"
        )


# ============================================================
# PHASE 2
# ============================================================

print()
print("=" * 70)
print("PHASE 2 - FINE-TUNE RESNET50 LAYER4")
print("=" * 70)
print()

model.unfreeze_layer4()

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.backbone[7].parameters(),
            "lr": LEARNING_RATE_LAYER4,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not name.startswith("backbone.7.")
                and parameter.requires_grad
            ],
            "lr": LEARNING_RATE_HEAD,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)

print("ResNet50 layers 1-3: FROZEN")
print("ResNet50 layer4: TRAINABLE")
print("Custom Conv + Linear layers: TRAINABLE")
print()

for epoch in range(1, PHASE2_EPOCHS + 1):

    train_loss, train_acc, train_macro_f1, train_weighted_f1 = (
        train_one_epoch(
            model,
            train_loader,
            optimizer,
        )
    )

    val_loss, val_acc, val_macro_f1, val_weighted_f1 = (
        validate(
            model,
            val_loader,
        )
    )

    epoch_data = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_macro_f1": train_macro_f1,
        "train_weighted_f1": train_weighted_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1,
        "val_weighted_f1": val_weighted_f1,
    }

    history["phase2"].append(epoch_data)

    print(
        f"Phase 2 | Epoch "
        f"{epoch:02d}/{PHASE2_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Train Macro F1: {train_macro_f1:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Macro F1: {val_macro_f1:.4f} | "
        f"Val Weighted F1: {val_weighted_f1:.4f}"
    )

    if val_macro_f1 > best_macro_f1:

        best_macro_f1 = val_macro_f1
        best_epoch = epoch
        best_phase = 2

        save_checkpoint(
            model,
            epoch,
            phase=2,
            val_accuracy=val_acc,
            val_macro_f1=val_macro_f1,
            val_weighted_f1=val_weighted_f1,
        )

        print(
            f"  ✓ New best model saved "
            f"(Macro F1: {val_macro_f1:.4f})"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history["best_model"] = {
    "phase": best_phase,
    "epoch": best_epoch,
    "val_macro_f1": best_macro_f1,
    "checkpoint": str(BEST_MODEL_PATH),
}

V4_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        history,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("V4 TRAINING COMPLETE")
print("=" * 70)
print()

print(f"Best Phase:        {best_phase}")
print(f"Best Epoch:        {best_epoch}")
print(f"Best Val Macro F1: {best_macro_f1:.4f}")
print()

print("Best model:")
print(f"  {BEST_MODEL_PATH}")

print()
print("Training history:")
print(f"  {HISTORY_PATH}")

print()
print("Training finished successfully.")