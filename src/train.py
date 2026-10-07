import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score

from src.model import NepaliDressClassifier


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODEL_DIR = "models"

BATCH_SIZE = 8

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 15

IMAGE_SIZE = 224

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_LAYER4 = 1e-5
LEARNING_RATE_FC = 1e-4

SEED = 42

# Use a limited number of CPU threads so the laptop remains responsive.
CPU_THREADS = 4


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(CPU_THREADS)

device = torch.device("cpu")

print("=" * 60)
print("Nepali Cultural Dress Recognition")
print("CPU Training - ResNet50")
print("=" * 60)

print(f"Device: {device}")
print(f"CPU threads: {CPU_THREADS}")
print(f"Batch size: {BATCH_SIZE}")
print()


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    ),
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    ),
])


# ============================================================
# DATASETS
# ============================================================

print("Loading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

class_names = train_dataset.classes
num_classes = len(class_names)

print(f"Number of classes: {num_classes}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print()

print("Classes:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")

print()


# ============================================================
# CHECK CLASS CONSISTENCY
# ============================================================

if train_dataset.classes != val_dataset.classes:
    raise RuntimeError(
        "Train and validation class folders do not match."
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("Calculating class weights...")

targets = np.array(train_dataset.targets)

class_counts = np.bincount(
    targets,
    minlength=num_classes
)

# Inverse square-root weighting is safer than
# inverse-frequency weighting for highly imbalanced data.
class_weights = 1.0 / np.sqrt(
    np.maximum(class_counts, 1)
)

# Normalize weights so their mean is approximately 1.
class_weights = class_weights / class_weights.mean()

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)

print("Class counts:")
for index, count in enumerate(class_counts):
    print(f"{index}: {class_names[index]} -> {count}")

print()


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# MODEL
# ============================================================

print("Creating ResNet50 model...")

model = NepaliDressClassifier(
    num_classes=num_classes
)

model = model.to(device)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    all_predictions = []
    all_targets = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_targets.extend(
            labels.detach().cpu().numpy()
        )

    epoch_loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return epoch_loss, accuracy, macro_f1


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                labels.cpu().numpy()
            )

    epoch_loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1,
        weighted_f1
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    epoch,
    val_accuracy,
    val_macro_f1,
    phase
):

    checkpoint = {
        "artifact_version": "1.0",
        "model_name": "resnet50",
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes,
        "image_size": IMAGE_SIZE,
        "mean": imagenet_mean,
        "std": imagenet_std,
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "phase": phase,
    }

    path = os.path.join(
        MODEL_DIR,
        "best_resnet50.pth"
    )

    torch.save(
        checkpoint,
        path
    )

    print(f"Saved best model -> {path}")


# ============================================================
# PHASE 1
# ============================================================

print("=" * 60)
print("PHASE 1")
print("Training classification head only")
print("=" * 60)

model.freeze_backbone()

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE_HEAD,
    weight_decay=1e-4
)

best_macro_f1 = -1.0

history = []


for epoch in range(PHASE1_EPOCHS):

    print()
    print(
        f"Phase 1 - Epoch "
        f"{epoch + 1}/{PHASE1_EPOCHS}"
    )

    train_loss, train_acc, train_f1 = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_acc, val_f1, val_weighted_f1 = validate(
        model,
        val_loader,
        criterion
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Train F1: {train_f1:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Macro F1: {val_f1:.4f} | "
        f"Val Weighted F1: {val_weighted_f1:.4f}"
    )

    history.append({
        "phase": 1,
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_macro_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "val_weighted_f1": val_weighted_f1,
    })

    if val_f1 > best_macro_f1:

        best_macro_f1 = val_f1

        save_checkpoint(
            model,
            epoch + 1,
            val_acc,
            val_f1,
            phase=1
        )


# ============================================================
# PHASE 2
# ============================================================

print()
print("=" * 60)
print("PHASE 2")
print("Fine-tuning ResNet50 layer4 + classifier")
print("=" * 60)

# Safely unfreeze layer4 and fc. Some model implementations may define a
# non-callable attribute with the same name (for example a tensor), so guard
# the method call explicitly and fall back to direct parameter updates.
unfreeze_layer4 = getattr(model, "unfreeze_layer4", None)
if callable(unfreeze_layer4) and not isinstance(unfreeze_layer4, torch.Tensor):
    unfreeze_layer4()
else:
    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad = True
    if hasattr(model, "backbone") and hasattr(model.backbone, "fc"):
        for parameter in model.backbone.fc.parameters():
            parameter.requires_grad = True

layer4_parameters = list(
    model.backbone.layer4.parameters()
)

fc_parameters = list(
    model.backbone.fc.parameters()
)

optimizer = torch.optim.AdamW(
    [
        {
            "params": layer4_parameters,
            "lr": LEARNING_RATE_LAYER4
        },
        {
            "params": fc_parameters,
            "lr": LEARNING_RATE_FC
        },
    ],
    weight_decay=1e-4
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)


for epoch in range(PHASE2_EPOCHS):

    print()
    print(
        f"Phase 2 - Epoch "
        f"{epoch + 1}/{PHASE2_EPOCHS}"
    )

    train_loss, train_acc, train_f1 = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_acc, val_f1, val_weighted_f1 = validate(
        model,
        val_loader,
        criterion
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Train F1: {train_f1:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Macro F1: {val_f1:.4f} | "
        f"Val Weighted F1: {val_weighted_f1:.4f}"
    )

    history.append({
        "phase": 2,
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_macro_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "val_weighted_f1": val_weighted_f1,
    })

    if val_f1 > best_macro_f1:

        best_macro_f1 = val_f1

        save_checkpoint(
            model,
            epoch + 1,
            val_acc,
            val_f1,
            phase=2
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    MODEL_DIR,
    "training_history.json"
)

with open(
    history_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation Macro F1: "
    f"{best_macro_f1:.4f}"
)

print(
    f"Best model: "
    f"models/best_resnet50.pth"
)

print(
    f"History: "
    f"models/training_history.json"
)

print("=" * 60)
