import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class NepaliDressClassifier(nn.Module):

    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.4,
        pretrained: bool = True,
    ):
        super().__init__()

        # ====================================================
        # RESNET50 WEIGHTS
        # ====================================================
        #
        # Training:
        # pretrained=True
        # -> use ImageNet pretrained weights
        #
        # Deployment:
        # pretrained=False
        # -> do NOT download ImageNet weights
        #
        # The trained checkpoint from S3 is loaded immediately
        # after the model architecture is created.
        # ====================================================

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = resnet50(
            weights=weights
        )

        # ====================================================
        # CLASSIFICATION HEAD
        # ====================================================

        in_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Sequential(
            nn.Dropout(
                dropout_rate
            ),

            nn.Linear(
                in_features,
                512
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                512
            ),

            nn.Dropout(
                dropout_rate / 2
            ),

            nn.Linear(
                512,
                num_classes
            ),
        )

    # ========================================================
    # PHASE 1
    # ========================================================

    def freeze_backbone(self):

        """
        Phase 1:
        Train only the classification head.
        """

        for param in self.backbone.parameters():

            param.requires_grad = False

        for param in self.backbone.fc.parameters():

            param.requires_grad = True

    # ========================================================
    # PHASE 2
    # ========================================================

    def unfreeze_layer4(self):

        """
        Phase 2:
        Fine-tune ResNet50 layer4 + classification head.
        """

        for param in self.backbone.parameters():

            param.requires_grad = False

        for param in self.backbone.layer4.parameters():

            param.requires_grad = True

        for param in self.backbone.fc.parameters():

            param.requires_grad = True

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        return self.backbone(x)