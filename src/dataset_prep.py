import json
import os
from pathlib import Path


DATASET_ROOT = Path(
    os.getenv("DATASET_ROOT", "data")
)

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_classes(split_dir):
    """Return class folder names in a dataset split."""
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {split_dir}"
        )

    return sorted(
        folder.name
        for folder in split_dir.iterdir()
        if folder.is_dir()
    )


def count_images(class_dir):
    """Count supported image files inside a class directory."""
    return sum(
        1
        for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_split(split):
    """Validate one dataset split and return image statistics."""
    split_dir = DATASET_ROOT / split
    classes = get_classes(split_dir)

    statistics = {}

    for class_name in classes:
        class_dir = split_dir / class_name
        statistics[class_name] = count_images(class_dir)

    return classes, statistics


def save_classes(class_names):
    """Save class mapping for reproducible inference."""
    output_path = DATASET_ROOT / "classes.json"

    data = {
        "num_classes": len(class_names),
        "classes": {
            str(index): name
            for index, name in enumerate(class_names)
        },
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nClass mapping saved to: {output_path}")


def main():
    print("=" * 60)
    print("NEPALI CULTURAL DRESS")
    print("CLASSIFICATION DATASET VALIDATION")
    print("=" * 60)

    print(f"\nDataset root: {DATASET_ROOT.resolve()}")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {DATASET_ROOT}"
        )

    split_data = {}

    for split in SPLITS:
        print(f"\nChecking {split.upper()} dataset...")

        classes, statistics = validate_split(split)

        split_data[split] = {
            "classes": classes,
            "statistics": statistics,
        }

        total_images = sum(statistics.values())

        print(f"Classes: {len(classes)}")
        print(f"Images: {total_images}")

    train_classes = split_data["train"]["classes"]

    print("\n" + "=" * 60)
    print("CLASS CONSISTENCY CHECK")
    print("=" * 60)

    for split in ["val", "test"]:
        current_classes = split_data[split]["classes"]

        missing = sorted(
            set(train_classes) - set(current_classes)
        )

        extra = sorted(
            set(current_classes) - set(train_classes)
        )

        if missing:
            print(
                f"\n{split.upper()} missing classes:"
            )
            for name in missing:
                print(f"  - {name}")

        if extra:
            print(
                f"\n{split.upper()} has unexpected classes:"
            )
            for name in extra:
                print(f"  - {name}")

        if not missing and not extra:
            print(
                f"{split.upper()}: OK"
            )

        if split == "val" and split_data["val"]["classes"] != train_classes:
            raise RuntimeError(
                "Train and validation class folders are inconsistent."
            )

    test_missing = sorted(
        set(train_classes) - set(split_data["test"]["classes"])
    )

    test_extra = sorted(
        set(split_data["test"]["classes"]) - set(train_classes)
    )

    if test_missing:
        print(
            "\nWARNING: Test split is missing classes:"
        )

        for class_name in test_missing:
            print(f"  - {class_name}")

    if test_extra:
        raise RuntimeError(
            "Test dataset contains classes that are not present "
            "in the training dataset."
        )

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    for index, class_name in enumerate(train_classes):
        train_count = split_data["train"]["statistics"][class_name]
        val_count = split_data["val"]["statistics"][class_name]
        test_count = split_data["test"]["statistics"].get(class_name, 0)

        print(
            f"{index:2d} | "
            f"{class_name:30s} | "
            f"train={train_count:4d} | "
            f"val={val_count:4d} | "
            f"test={test_count:4d}"
        )

    empty_classes = []

    for split in SPLITS:
        for class_name, count in split_data[split]["statistics"].items():
            if count == 0:
                empty_classes.append(
                    f"{split}/{class_name}"
                )

    if empty_classes:
        print("\nWARNING: Empty class folders found:")

        for item in empty_classes:
            print(f"  - {item}")

        raise RuntimeError(
            "One or more class folders contain no images."
        )

    save_classes(train_classes)

    print("\n" + "=" * 60)
    print("DATASET VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
