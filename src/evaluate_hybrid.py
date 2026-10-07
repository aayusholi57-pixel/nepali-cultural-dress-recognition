from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.hybrid_model import NepaliDressHybridClassifier


DATA_DIR = Path("data")
MODEL_PATH = Path("models/hybrid/best_hybrid_v1.pth")
OUTPUT_DIR = Path("reports/hybrid")

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint["mean"],
            std=checkpoint["std"],
        ),
    ])

    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=transform,
    )

    if test_dataset.classes != class_names:
        print(
            "Warning: test class folders differ from the "
            "training class mapping."
        )

    model = NepaliDressHybridClassifier(
        num_classes=num_classes,
        pretrained=False,
    ).to(device)

    model.load_state_dict(
        checkpoint["state_dict"]
    )
    model.eval()

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(
                predictions.cpu().numpy().tolist()
            )

    accuracy = accuracy_score(y_true, y_pred)
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

    labels_present = sorted(
        set(y_true) | set(y_pred)
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_present,
        target_names=[
            class_names[i]
            for i in labels_present
        ],
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels_present,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUTPUT_DIR / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    import numpy as np

    np.savetxt(
        OUTPUT_DIR / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )

    print("\nHybrid V1 Test Results")
    print("----------------------")
    print(f"Test images: {len(test_dataset)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Best training epoch: {checkpoint['epoch']}")
    print(
        f"Validation Accuracy: "
        f"{checkpoint['val_accuracy']:.4f}"
    )
    print(
        f"Validation Macro F1: "
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    print("\nClassification Report")
    print(report)

    print(
        f"\nSaved report files to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
