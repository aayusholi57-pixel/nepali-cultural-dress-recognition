import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.model import NepaliDressClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
MODEL_DIR = os.path.join("models", "resnet50")

IMAGE_SIZE = 224
BATCH_SIZE = 8

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 15

# Stage 1: classifier only
STAGE1_LR = 0.001

# Stage 2: different learning rates
CLASSIFIER_LR = 0.0001
BACKBONE_LR = 0.00001

WEIGHT_DECAY = 0.01

SEED = 42

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_resnet50.pth"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================
# TRAINING TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.0)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


# ============================================================
# VALIDATION / TEST TRANSFORMS
# ============================================================

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


# ============================================================
# DATASETS
# ============================================================

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

os.makedirs(MODEL_DIR, exist_ok=True)

for dataset_dir, label in [(train_dir, "training"), (val_dir, "validation")]:
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"{label.capitalize()} dataset directory not found: {dataset_dir}")

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=eval_transform
)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = train_dataset.classes
num_classes = len(class_names)

print(f"Classes: {num_classes}")
print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"  {index} -> {class_name}")


# ============================================================
# CHECK CLASS MAPPING
# ============================================================

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError("Train and validation class mappings do not match.")


# ============================================================
# DATASET INFORMATION
# ============================================================

print(f"\nTraining images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# CREATE MODEL
# ============================================================

model = NepaliDressClassifier(
    num_classes=num_classes,
    pretrained=True
)
model = model.to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    train_dataset.targets,
    minlength=num_classes
).astype(np.float32)

print("\nClass counts:")
for index, count in enumerate(class_counts):
    print(f"  {class_names[index]}: {count:.0f}")

safe_class_counts = np.where(class_counts == 0, 1.0, class_counts)
class_weights = len(train_dataset) / (num_classes * safe_class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

print("\nClass weights:")
for index, weight in enumerate(class_weights):
    print(f"  {class_names[index]}: {weight.item():.3f}")


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(weight=class_weights)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_predictions = []
    all_targets = []

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, targets in progress_bar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_accuracy = accuracy_score(all_targets, all_predictions)
    epoch_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return epoch_loss, epoch_accuracy, epoch_f1


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validation", leave=False)

        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_accuracy = accuracy_score(all_targets, all_predictions)
    epoch_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return epoch_loss, epoch_accuracy, epoch_f1


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(model, epoch, val_accuracy, val_macro_f1, phase):
    os.makedirs(MODEL_DIR, exist_ok=True)

    checkpoint = {
        "artifact_version": "2.0",
        "model_name": "resnet50_nepali_cultural_dress",
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes,
        "image_size": IMAGE_SIZE,
        "mean": MEAN,
        "std": STD,
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "phase": phase,
    }

    torch.save(checkpoint, BEST_MODEL_PATH)
    print(f"\nSaved best model -> {BEST_MODEL_PATH}")


# ============================================================
# STAGE 1
# ============================================================
# Freeze ResNet50 backbone.
# Train only the new classification head.
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1: TRAIN CLASSIFIER HEAD")
print("=" * 70)

model.freeze_backbone()

trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=STAGE1_LR,
    weight_decay=WEIGHT_DECAY
)

best_val_f1 = -1.0

for epoch in range(1, STAGE1_EPOCHS + 1):
    print(f"\nStage 1 Epoch {epoch}/{STAGE1_EPOCHS}")

    train_loss, train_acc, train_f1 = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )

    val_loss, val_acc, val_f1 = validate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    print(f"\nTrain Loss: {train_loss:.4f}")
    print(f"Train Acc:  {train_acc:.4f}")
    print(f"Train F1:   {train_f1:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Acc:    {val_acc:.4f}")
    print(f"Val F1:     {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        save_best_model(
            model=model,
            epoch=epoch,
            val_accuracy=val_acc,
            val_macro_f1=val_f1,
            phase="stage1"
        )


# ============================================================
# STAGE 2
# ============================================================
# Unfreeze the complete ResNet50.
# Use separate learning rates for backbone and classifier.
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2: FULL RESNET50 FINE-TUNING")
print("=" * 70)

model.unfreeze_all()

backbone_parameters = [
    parameter
    for name, parameter in model.backbone.named_parameters()
    if not name.startswith("fc.")
]

classifier_parameters = model.backbone.fc.parameters()

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_parameters, "lr": BACKBONE_LR},
        {"params": classifier_parameters, "lr": CLASSIFIER_LR},
    ],
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=STAGE2_EPOCHS
)

for epoch in range(1, STAGE2_EPOCHS + 1):
    print(f"\nStage 2 Epoch {epoch}/{STAGE2_EPOCHS}")

    train_loss, train_acc, train_f1 = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )

    val_loss, val_acc, val_f1 = validate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    print(f"\nTrain Loss: {train_loss:.4f}")
    print(f"Train Acc:  {train_acc:.4f}")
    print(f"Train F1:   {train_f1:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Acc:    {val_acc:.4f}")
    print(f"Val F1:     {val_f1:.4f}")

    print(f"Backbone LR: {optimizer.param_groups[0]['lr']:.8f}")
    print(f"Classifier LR: {optimizer.param_groups[1]['lr']:.8f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        save_best_model(
            model=model,
            epoch=epoch,
            val_accuracy=val_acc,
            val_macro_f1=val_f1,
            phase="stage2"
        )

    scheduler.step()


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"\nBest validation Macro F1: {best_val_f1:.4f}")
print("Best model saved at:")
print(BEST_MODEL_PATH)

print("\nOriginal model was NOT overwritten.")
print(f"Next step: evaluate {BEST_MODEL_PATH} on the test dataset.")
