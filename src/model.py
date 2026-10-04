import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class NepaliDressClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.4):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT
        self.backbone = resnet50(weights=weights)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

    def freeze_backbone(self):
        """Phase 1: train only the classification head."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_layer4(self):
        """Phase 2: fine-tune only ResNet layer4 + classification head."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
