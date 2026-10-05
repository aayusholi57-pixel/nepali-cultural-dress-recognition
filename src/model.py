import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class NepaliDressClassifier(nn.Module):
    """
    ResNet50 model for Nepali cultural dress classification.
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # Load ResNet50 with ImageNet knowledge
        weights = ResNet50_Weights.DEFAULT if pretrained else None

        self.backbone = resnet50(weights=weights)

        # Number of features coming from ResNet50
        input_features = self.backbone.fc.in_features

        # Original trained classifier architecture.
        # ResNet's `fc` is typed as a Linear layer, but we intentionally replace it
        # with a custom sequential head for the classification task.
        self.backbone.fc = nn.Sequential(  # type: ignore[assignment]
            nn.Dropout(0.4),
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def freeze_backbone(self):
        """
        Freeze ResNet50 feature extractor.
        Train only the classifier.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

    def unfreeze_all(self):
        """
        Unfreeze the complete ResNet50.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(self, x):
        return self.backbone(x)