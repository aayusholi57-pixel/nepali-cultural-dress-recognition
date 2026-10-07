import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class NepaliDressHybridClassifier(nn.Module):
    """
    Hybrid classifier for Nepali cultural dress recognition.

    Uses only the early/mid feature extraction part of pretrained
    ResNet50 (Conv1 + Layer1 + Layer2). Layer3, Layer4, the original
    ResNet50 pooling layer, and the original fully connected layer
    are intentionally not used.

    Task-specific feature learning is performed by custom Conv2D
    and Linear layers.
    """

    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.4,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # Keep only the feature extraction required for this model.
        self.resnet_features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )

        # Do not use ResNet50 Layer3, Layer4, avgpool, or fc.
        for param in self.resnet_features.parameters():
            param.requires_grad = False

        # Custom convolutional feature extractor.
        self.custom_cnn = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Convert spatial features to one feature vector.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom classification head.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def freeze_resnet(self):
        """Keep all selected pretrained ResNet features frozen."""
        for param in self.resnet_features.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        """Return only parameters belonging to the custom model."""
        return (
            param
            for param in self.parameters()
            if param.requires_grad
        )

    def forward(self, x):
        x = self.resnet_features(x)
        x = self.custom_cnn(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def feature_shape(self, image_size=224):
        """Return the feature-map shape after the selected ResNet part."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            features = self.resnet_features(dummy)
        return tuple(features.shape)
