import io
import os
from typing import cast

import boto3
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.v4_model import ResNet50CustomV4


# ============================================================
# CONFIG
# ============================================================

S3_BUCKET = os.getenv(
    "S3_BUCKET",
    "nepali-cultural-dress-ai-696822062401-ap-southeast-2-an",
)

S3_MODEL_KEY = os.getenv(
    "S3_MODEL_KEY",
    "models/resnet50_custom_v4/best_resnet50_custom_v4.pth",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CONFIDENCE_THRESHOLD = 0.60

IMAGE_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MAX_IMAGE_SIZE = 5 * 1024 * 1024


# ============================================================
# PREDICTOR SERVICE
# ============================================================

class PredictorService:

    def __init__(self):

        self.model = None
        self.class_names = []
        self.num_classes = 0
        self.checkpoint_info = {}

        self.transform = transforms.Compose([
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
    # CREATE S3 CLIENT
    # ========================================================

    def _get_s3_client(self):
        region = (
            os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
        )
        profile_name = os.getenv("AWS_PROFILE")

        if profile_name:
            try:
                session = boto3.Session(
                    profile_name=profile_name,
                    region_name=region,
                )
                return session.client("s3")
            except Exception as exc:
                print(
                    f"[WARNING] Could not load AWS_PROFILE '{profile_name}': {exc}"
                )
                print(
                    "[INFO] Falling back to default AWS credential chain..."
                )

        return boto3.client("s3", region_name=region)

    # ========================================================
    # LOAD MODEL FROM S3
    # ========================================================

    def load_model_from_s3(self):

        print("=" * 60)
        print("LOADING RESNET50 CUSTOM V4 MODEL")
        print("=" * 60)

        print(
            f"Device: {DEVICE}"
        )

        print(
            f"S3 Bucket: {S3_BUCKET}"
        )

        print(
            f"S3 Model: {S3_MODEL_KEY}"
        )

        print()

        try:

            s3 = self._get_s3_client()

            buffer = io.BytesIO()

            print(
                "Downloading model from S3..."
            )

            s3.download_fileobj(
                S3_BUCKET,
                S3_MODEL_KEY,
                buffer
            )

            buffer.seek(0)

            print(
                "Model download completed."
            )

            # ------------------------------------------------
            # Load checkpoint
            # ------------------------------------------------

            checkpoint = torch.load(
                buffer,
                map_location=DEVICE,
                weights_only=False
            )

            # ------------------------------------------------
            # Read metadata
            # ------------------------------------------------

            self.class_names = checkpoint[
                "class_names"
            ]

            self.num_classes = checkpoint[
                "num_classes"
            ]

            self.checkpoint_info = {
                "model_name": checkpoint.get(
                    "model_name",
                    "resnet50"
                ),
                "artifact_version": checkpoint.get(
                    "artifact_version",
                    "unknown"
                ),
                "epoch": checkpoint.get(
                    "epoch"
                ),
                "val_accuracy": checkpoint.get(
                    "val_accuracy"
                ),
                "val_macro_f1": checkpoint.get(
                    "val_macro_f1"
                ),
            }

            # ------------------------------------------------
            # Create model
            # ------------------------------------------------

            self.model = ResNet50CustomV4(
                num_classes=self.num_classes,
                pretrained=False,
            )

            # ------------------------------------------------
            # Load trained weights
            # ------------------------------------------------

            self.model.load_state_dict(
                checkpoint["state_dict"]
            )

            self.model.to(
                DEVICE
            )

            self.model.eval()

            print()
            print(
                "Model loaded successfully."
            )

            print(
                f"Number of classes: "
                f"{self.num_classes}"
            )

            print()

            print(
                "Supported classes:"
            )

            for index, class_name in enumerate(
                self.class_names
            ):

                print(
                    f"  {index:2d} -> {class_name}"
                )

            print()

        except Exception as exc:

            self.model = None

            print()
            print("ERROR: Failed to load model.")
            if exc.__class__.__name__ == "NoCredentialsError":
                print(
                    "Reason: AWS credentials are missing. Attach an EC2 IAM role "
                    "or provide AWS credentials through the runtime environment."
                )
            else:
                print(f"Reason: {exc}")

            raise

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        image_bytes: bytes,
        top_k: int = 3
    ):

        if self.model is None:

            raise RuntimeError(
                "Model is not loaded."
            )

        if not image_bytes:

            raise ValueError(
                "Empty image file."
            )

        if len(image_bytes) > MAX_IMAGE_SIZE:

            raise ValueError(
                "Image exceeds the 5 MB size limit."
            )

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        try:

            loaded_image = Image.open(
                io.BytesIO(image_bytes)
            )

            loaded_image = loaded_image.convert(
                "RGB"
            )

        except UnidentifiedImageError:

            raise ValueError(
                "The uploaded file is not a valid image."
            )

        except Exception:

            raise ValueError(
                "Unable to read the uploaded image."
            )

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        input_tensor = cast(
            torch.Tensor,
            self.transform(loaded_image),
        )
        input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.model(input_tensor)

            probabilities = F.softmax(
                outputs,
                dim=1
            )[0]

        # ----------------------------------------------------
        # Top-K predictions
        # ----------------------------------------------------

        k = min(
            max(top_k, 1),
            len(self.class_names)
        )

        top_probs, top_indices = torch.topk(
            probabilities,
            k=k
        )

        predictions = []

        for probability, index in zip(
            top_probs,
            top_indices
        ):

            class_index = int(
                index.item()
            )

            confidence = (
                probability.item() * 100
            )

            predictions.append(
                {
                    "class_name": (
                        self.class_names[
                            class_index
                        ]
                    ),
                    "confidence": round(
                        confidence,
                        2
                    ),
                }
            )

        # ----------------------------------------------------
        # Top prediction
        # ----------------------------------------------------

        top_confidence = (
            predictions[0]["confidence"] / 100.0
        )

        # ----------------------------------------------------
        # Confidence-based uncertainty
        # ----------------------------------------------------

        is_ood = (
            top_confidence
            < CONFIDENCE_THRESHOLD
        )

        return predictions, is_ood


# ============================================================
# GLOBAL PREDICTOR SERVICE
# ============================================================

predictor_service = PredictorService()
