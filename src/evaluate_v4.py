import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.v4_model import ResNet50CustomV4


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data")
MODEL_PATH = Path(
    "models/resnet50_custom_v4/best_resnet50_custom_v4.pth"
)
CLASSES_PATH = DATA_DIR / "classes.json"

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("V4 MODEL TEST EVALUATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V4 checkpoint not found: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load class names
    # --------------------------------------------------------

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_data = json.load(f)

    class_names = [
    class_data["classes"][str(i)]
    for i in range(class_data["num_classes"])
]

    print(f"Number of classes: {len(class_names)}")
    print("Classes:")

    for index, class_name in enumerate(class_names):
        print(f"  {index:2d} -> {class_name}")

    # --------------------------------------------------------
    # Test transforms
    # --------------------------------------------------------

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_dir = DATA_DIR / "test"

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=test_transform,
    )

    print()
    print(f"Test images: {len(test_dataset)}")
    print(f"Test classes found: {len(test_dataset.classes)}")

    print("Test classes:")
    for index, class_name in enumerate(test_dataset.classes):
        print(f"  {index:2d} -> {class_name}")

    # --------------------------------------------------------
    # Test class -> model class mapping
    #
    # The test dataset does not contain every training class.
    # Therefore ImageFolder's test labels cannot be used
    # directly as model labels.
    # --------------------------------------------------------

    test_index_to_model_index = {
        test_index: class_names.index(class_name)
        for test_index, class_name in enumerate(test_dataset.classes)
    }

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = ResNet50CustomV4(
        num_classes=len(class_names),
        pretrained=False,
    )

    checkpoint = torch.load(

        MODEL_PATH,
        map_location=DEVICE,
    )

    # V4 checkpoint stores the actual model weights
    # inside the "state_dict" key.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print("Model checkpoint loaded successfully.")

    model.to(DEVICE)
    model.eval()

    print()
    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            for label in labels.numpy():
                y_true.append(
                    test_index_to_model_index[int(label)]
                )

            y_pred.extend(
                predictions.cpu().numpy().tolist()
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        average="weighted",
        zero_division=0,
    )

    print()
    print("=" * 70)
    print("V4 TEST RESULTS")
    print("=" * 70)

    print(f"Accuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro F1:    {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
    print(f"Weighted F1: {weighted_f1:.4f} ({weighted_f1 * 100:.2f}%)")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(cm)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "model": "ResNet50CustomV4",
        "checkpoint": str(MODEL_PATH),
        "test_images": len(test_dataset),
        "num_classes": len(class_names),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "class_names": class_names,
        "test_classes": test_dataset.classes,
        "confusion_matrix": cm.tolist(),
    }

    output_path = Path(
        "models/resnet50_custom_v4/test_results_v4.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print(f"Results saved to: {output_path}")

    print()
    print("=" * 70)
    print("V4 EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()