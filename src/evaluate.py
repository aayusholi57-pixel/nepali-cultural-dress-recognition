import json
import shutil
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from src.model import NepaliDressClassifier


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")
TEST_DIR = DATA_DIR / "test"

MODEL_PATH = Path("models/best_resnet50.pth")

OUTPUT_DIR = Path("evaluation")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    print("=" * 60)
    print("NEPALI CULTURAL DRESS - RESNET50 EVALUATION")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_PATH}")
    print()

    # ========================================================
    # 1. CHECK MODEL
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run training first."
        )

    if not TEST_DIR.exists():

        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )

    # ========================================================
    # 2. LOAD CHECKPOINT
    # ========================================================

    print("Loading checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    print(
        f"Number of classes: {num_classes}"
    )

    print()

    # ========================================================
    # 3. CREATE MODEL
    # ========================================================

    model = NepaliDressClassifier(
        num_classes=num_classes
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")
    print()

    # ========================================================
    # 4. TRANSFORMS
    # ========================================================

    eval_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            MEAN,
            STD
        ),
    ])

    # ========================================================
    # 5. FIND TEST CLASSES
    # ========================================================

    print("Checking test dataset...")

    available_classes = []
    missing_classes = []

    for class_name in class_names:

        class_dir = TEST_DIR / class_name

        if not class_dir.exists():

            missing_classes.append(
                class_name
            )

            continue

        image_files = [
            file
            for file in class_dir.iterdir()
            if (
                file.is_file()
                and file.suffix.lower()
                in VALID_EXTENSIONS
            )
        ]

        if len(image_files) > 0:

            available_classes.append(
                class_name
            )

        else:

            missing_classes.append(
                class_name
            )

    print(
        f"Classes with test images: "
        f"{len(available_classes)}"
    )

    print(
        f"Classes without test images: "
        f"{len(missing_classes)}"
    )

    if missing_classes:

        print(
            "Missing test classes:"
        )

        for class_name in missing_classes:

            print(
                f"  - {class_name}"
            )

    print()

    # ========================================================
    # 6. CREATE TEMPORARY TEST DATASET
    # ========================================================
    #
    # ImageFolder requires every class directory to contain
    # at least one image.
    #
    # Therefore we create a temporary directory containing
    # ONLY classes that actually have test images.
    #
    # IMPORTANT:
    # We later map ImageFolder indices back to the original
    # 24-class model indices.
    #

    temp_test_dir = (
        OUTPUT_DIR /
        "_test_for_evaluation"
    )

    if temp_test_dir.exists():

        shutil.rmtree(
            temp_test_dir
        )

    temp_test_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for class_name in available_classes:

        source_dir = (
            TEST_DIR /
            class_name
        )

        destination_dir = (
            temp_test_dir /
            class_name
        )

        try:

            # Windows supports directory symlinks,
            # but they may require special permissions.

            destination_dir.symlink_to(
                source_dir.resolve(),
                target_is_directory=True,
            )

        except OSError:

            # Fallback: copy the directory.

            shutil.copytree(
                source_dir,
                destination_dir,
            )

    # ========================================================
    # 7. CREATE IMAGEFOLDER DATASET
    # ========================================================

    test_dataset = datasets.ImageFolder(
        temp_test_dir,
        transform=eval_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Test images: {len(test_dataset)}"
    )

    print(
        "ImageFolder classes:"
    )

    for index, class_name in enumerate(
        test_dataset.classes
    ):

        print(
            f"  {index:2d} -> {class_name}"
        )

    print()

    # ========================================================
    # 8. PREDICTION
    # ========================================================

    y_true = []
    y_pred = []

    print(
        "Running predictions..."
    )

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    print(
        "Prediction completed."
    )

    print()

    # ========================================================
    # 9. FIX TEST LABEL INDICES
    # ========================================================
    #
    # IMPORTANT:
    #
    # The model was trained using all 24 classes.
    #
    # But ImageFolder contains only the 23 classes that have
    # test images.
    #
    # Because "naugedi" is missing from the test set,
    # ImageFolder's indices after "naugedi" are shifted.
    #
    # Example:
    #
    # MODEL:
    # naugedi = 16
    # patuki  = 17
    #
    # TEST ImageFolder:
    # patuki  = 16
    #
    # Therefore we map ImageFolder index -> model index.
    #

    test_index_to_model_index = {
        test_index: class_names.index(
            class_name
        )
        for test_index, class_name
        in enumerate(
            test_dataset.classes
        )
    }

    y_true = [
        test_index_to_model_index[
            index
        ]
        for index in y_true
    ]

    print(
        "Test labels remapped to model "
        "class indices."
    )

    print()

    # ========================================================
    # 10. CLASSIFICATION REPORT
    # ========================================================

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(num_classes)
        ),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(
                range(num_classes)
            ),
            target_names=class_names,
            zero_division=0,
        )
    )

    # ========================================================
    # 11. SAVE METRICS
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_path = (
        OUTPUT_DIR /
        "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"Metrics saved to: "
        f"{metrics_path}"
    )

    # ========================================================
    # 12. CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(num_classes)
        ),
    )

    plt.figure(
        figsize=(18, 16)
    )

    plt.imshow(
        cm,
        interpolation="nearest",
    )

    plt.title(
        "Confusion Matrix - ResNet50"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        range(num_classes),
        class_names,
        rotation=90,
    )

    plt.yticks(
        range(num_classes),
        class_names,
    )

    # --------------------------------------------------------
    # Add numbers to confusion matrix
    # --------------------------------------------------------

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            value = cm[i, j]

            if value != 0:

                plt.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                )

    plt.colorbar()

    plt.tight_layout()

    confusion_path = (
        OUTPUT_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Confusion matrix saved to: "
        f"{confusion_path}"
    )

    # ========================================================
    # 13. SUMMARY METRICS
    # ========================================================

    accuracy = report.get(
        "accuracy",
        0.0
    )

    macro_f1 = report["macro avg"][
        "f1-score"
    ]

    weighted_f1 = report[
        "weighted avg"
    ][
        "f1-score"
    ]

    print()

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Accuracy:    "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1:    "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted F1: "
        f"{weighted_f1 * 100:.2f}%"
    )

    print("=" * 60)

    # ========================================================
    # 14. MODEL INFORMATION
    # ========================================================

    print()
    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)

    if "epoch" in checkpoint:

        print(
            f"Best Epoch: "
            f"{checkpoint['epoch']}"
        )

    if "phase" in checkpoint:

        print(
            f"Best Phase: "
            f"{checkpoint['phase']}"
        )

    if "val_accuracy" in checkpoint:

        print(
            f"Best Validation Accuracy: "
            f"{checkpoint['val_accuracy'] * 100:.2f}%"
        )

    if "val_macro_f1" in checkpoint:

        print(
            f"Best Validation Macro F1: "
            f"{checkpoint['val_macro_f1'] * 100:.2f}%"
        )

    print("=" * 60)

    # ========================================================
    # 15. TEST DATA WARNING
    # ========================================================

    if missing_classes:

        print()
        print("=" * 60)
        print("TEST DATASET NOTE")
        print("=" * 60)

        print(
            "The following classes have no test images:"
        )

        for class_name in missing_classes:

            print(
                f"  - {class_name}"
            )

        print(
            "\nTheir precision/recall/F1 are shown as 0 "
            "because there are no ground-truth test "
            "samples for those classes."
        )

        print("=" * 60)

    # ========================================================
    # 16. CLEANUP
    # ========================================================

    if temp_test_dir.exists():

        shutil.rmtree(
            temp_test_dir,
            ignore_errors=True
        )

    print()
    print(
        "Temporary evaluation files cleaned."
    )

    print()
    print(
        "Evaluation completed successfully."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate()
