import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50CustomV4(nn.Module):
    """
    V4 model for Nepali Cultural Dress Recognition.

    Architecture:
        Input 224x224x3
            ↓
        Pretrained ResNet50 backbone
            ↓
        Feature map: 2048x7x7
            ↓
        Custom Conv2D: 2048 → 512
            ↓
        ReLU
            ↓
        Custom Conv2D: 512 → 256
            ↓
        ReLU
            ↓
        Global Average Pooling
            ↓
        256
            ↓
        Custom Linear: 256 → 128
            ↓
        ReLU
            ↓
        Dropout
            ↓
        Custom Linear: 128 → num_classes
    """

    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()

        # =====================================================
        # 1. PRETRAINED RESNET50 BACKBONE
        # =====================================================

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet50(weights=weights)

        # Keep the complete ResNet50 convolutional
        # feature extractor.
        #
        # We remove:
        #   - ResNet50 avgpool
        #   - ResNet50 original fc
        #
        # because we will build our own head.

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # =====================================================
        # 2. YOUR CUSTOM CONVOLUTIONAL LAYERS
        # =====================================================

        self.conv1 = nn.Conv2d(
            in_channels=2048,
            out_channels=512,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU(inplace=True)

        # =====================================================
        # 3. GLOBAL AVERAGE POOLING
        # =====================================================

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # =====================================================
        # 4. YOUR CUSTOM LINEAR LAYERS
        # =====================================================

        self.linear1 = nn.Linear(
            256,
            128,
        )

        self.dropout = nn.Dropout(
            p=dropout_rate
        )

        self.linear2 = nn.Linear(
            128,
            num_classes,
        )

    # =========================================================
    # FORWARD PASS
    # =========================================================

    def forward(self, x):

        # ResNet50 feature extraction
        x = self.backbone(x)

        # Custom convolutional layers
        x = self.relu(
            self.conv1(x)
        )

        x = self.relu(
            self.conv2(x)
        )

        # Global average pooling
        x = self.global_pool(x)

        # Flatten
        x = torch.flatten(
            x,
            start_dim=1,
        )

        # Custom linear layers
        x = self.relu(
            self.linear1(x)
        )

        x = self.dropout(x)

        x = self.linear2(x)

        return x

    # =========================================================
    # PHASE 1
    # =========================================================

    def freeze_backbone(self):
        """
        Phase 1:

        Freeze the complete pretrained ResNet50.

        Train only:
            - Custom Conv1
            - Custom Conv2
            - Custom Linear1
            - Custom Linear2
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.conv1.parameters():
            parameter.requires_grad = True

        for parameter in self.conv2.parameters():
            parameter.requires_grad = True

        for parameter in self.linear1.parameters():
            parameter.requires_grad = True

        for parameter in self.linear2.parameters():
            parameter.requires_grad = True

    # =========================================================
    # PHASE 2
    # =========================================================

    def unfreeze_layer4(self):
        """
        Phase 2:

        Keep ResNet50 layers 1-3 frozen.

        Unfreeze only ResNet50 layer4.

        Custom convolutional and linear layers
        remain trainable.
        """

        # Freeze complete backbone first.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # ResNet50 layer4 is index 7 in the Sequential backbone.
        for parameter in self.backbone[7].parameters():
            parameter.requires_grad = True

        # Keep all custom layers trainable.
        for module in [
            self.conv1,
            self.conv2,
            self.linear1,
            self.linear2,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    # =========================================================
    # TRAINABLE PARAMETERS
    # =========================================================

    def trainable_parameters(self):
        """
        Return only parameters that should currently
        be optimized.
        """

        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )