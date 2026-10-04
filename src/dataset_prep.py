from pathlib import Path
from PIL import Image
import random
import yaml
import json
import csv


DATASET_ROOT = Path(
    "/home/ubuntu/.cache/kagglehub/datasets/"
    "bimarshakhanal/nepali-cultural-dress-and-ornaments/"
    "versions/1"
)

OUTPUT_ROOT = Path("data")

VAL_RATIO = 0.15
SEED = 42

random.seed(SEED)


def load_classes():
    yaml_path = DATASET_ROOT / "data.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config["names"]


def find_images(directory):
    extensions = {".jpg", ".jpeg", ".png"}

    return sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in extensions
        ]
    )


def read_yolo_labels(label_path):
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            boxes.append(
                (
                    class_id,
                    x_center,
                    y_center,
                    width,
                    height,
                )
            )

    return boxes


def yolo_to_pixels(
    x_center,
    y_center,
    width,
    height,
    image_width,
    image_height,
):

    x1 = int(
        (x_center - width / 2) * image_width
    )

    y1 = int(
        (y_center - height / 2) * image_height
    )

    x2 = int(
        (x_center + width / 2) * image_width
    )

    y2 = int(
        (y_center + height / 2) * image_height
    )

    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    return x1, y1, x2, y2


def create_directories(class_names):

    for split in ["train", "val", "test"]:

        for class_name in class_names:

            (
                OUTPUT_ROOT
                / split
                / class_name
            ).mkdir(
                parents=True,
                exist_ok=True,
            )


def process_images(
    image_paths,
    source_split,
    output_split,
    class_names,
    metadata,
):

    image_dir = (
        DATASET_ROOT
        / source_split
        / "images"
    )

    label_dir = (
        DATASET_ROOT
        / source_split
        / "labels"
    )

    total_crops = 0
    total_images = 0

    for image_path in image_paths:

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        boxes = read_yolo_labels(label_path)

        if not boxes:
            continue

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            print(
                f"Skipping {image_path}: {e}"
            )

            continue

        image_width, image_height = image.size

        image_crops = 0

        for box_index, box in enumerate(boxes):

            (
                class_id,
                x_center,
                y_center,
                width,
                height,
            ) = box

            if (
                class_id < 0
                or class_id >= len(class_names)
            ):
                continue

            class_name = class_names[class_id]

            (
                x1,
                y1,
                x2,
                y2,
            ) = yolo_to_pixels(
                x_center,
                y_center,
                width,
                height,
                image_width,
                image_height,
            )

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop(
                (x1, y1, x2, y2)
            )

            if (
                crop.width < 10
                or crop.height < 10
            ):
                continue

            output_dir = (
                OUTPUT_ROOT
                / output_split
                / class_name
            )

            output_name = (
                f"{image_path.stem}"
                f"_box_{box_index}.jpg"
            )

            output_path = (
                output_dir
                / output_name
            )

            crop.save(
                output_path,
                "JPEG",
                quality=95,
            )

            metadata.append(
                {
                    "split": output_split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "source_image": image_path.name,
                    "crop_path": str(output_path),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

            total_crops += 1
            image_crops += 1

        if image_crops > 0:
            total_images += 1

    print(
        f"{output_split}: "
        f"{total_images} images -> "
        f"{total_crops} crops"
    )


def save_metadata(metadata):

    path = OUTPUT_ROOT / "metadata.csv"

    fields = [
        "split",
        "class_id",
        "class_name",
        "source_image",
        "crop_path",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(metadata)


def save_classes(class_names):

    path = OUTPUT_ROOT / "classes.json"

    data = {
        "num_classes": len(class_names),
        "classes": {
            str(i): name
            for i, name in enumerate(class_names)
        },
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def print_statistics(metadata, class_names):

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    for split in ["train", "val", "test"]:

        print(f"\n{split.upper()}")

        split_data = [
            item
            for item in metadata
            if item["split"] == split
        ]

        for class_id, class_name in enumerate(
            class_names
        ):

            count = sum(
                1
                for item in split_data
                if item["class_id"] == class_id
            )

            print(
                f"{class_id:2d} | "
                f"{class_name:30s} | "
                f"{count}"
            )


def main():

    print("=" * 60)
    print("NEPALI CULTURAL DRESS")
    print("YOLO -> RESNET50 CLASSIFICATION DATASET")
    print("=" * 60)

    class_names = load_classes()

    print(
        f"\nNumber of classes: "
        f"{len(class_names)}"
    )

    for i, name in enumerate(class_names):
        print(f"{i:2d} -> {name}")

    create_directories(class_names)

    train_image_dir = (
        DATASET_ROOT
        / "train"
        / "images"
    )

    test_image_dir = (
        DATASET_ROOT
        / "test"
        / "images"
    )

    train_images = find_images(
        train_image_dir
    )

    test_images = find_images(
        test_image_dir
    )

    print(
        f"\nOriginal train images: "
        f"{len(train_images)}"
    )

    print(
        f"Official test images: "
        f"{len(test_images)}"
    )

    random.shuffle(train_images)

    val_count = int(
        len(train_images) * VAL_RATIO
    )

    val_images = train_images[:val_count]

    new_train_images = train_images[val_count:]

    print(
        f"Training images: "
        f"{len(new_train_images)}"
    )

    print(
        f"Validation images: "
        f"{len(val_images)}"
    )

    metadata = []

    print("\nCreating crops...")

    process_images(
        new_train_images,
        "train",
        "train",
        class_names,
        metadata,
    )

    process_images(
        val_images,
        "train",
        "val",
        class_names,
        metadata,
    )

    process_images(
        test_images,
        "test",
        "test",
        class_names,
        metadata,
    )

    save_metadata(metadata)

    save_classes(class_names)

    print_statistics(
        metadata,
        class_names,
    )

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"\nOutput directory: "
        f"{OUTPUT_ROOT.resolve()}"
    )


if __name__ == "__main__":
    main()
