"""
cnn_model.py
------------
A compact convolutional neural network for classifying Mel-spectrograms (10 digits).

Architecture:
- 3 conv blocks with ReLU activations and downsampling
- Global average pooling to a 64-D embedding
- Linear classifier to 10 classes

Intended input shape:
    [batch, 1, n_mels, time]

Usage:
    from collab.zsolt.cnn_model import SmallCNN
    model = SmallCNN(n_classes=10)
"""

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()

        # Feature extractor: conv → relu → pool, repeated
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # halve both freq and time

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Pool to 1x1 regardless of time dimension length
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Classifier: maps pooled 64-dim features to class logits
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: [B, 1, n_mels, time]
        x = self.features(x)      # [B, 64, 1, 1]
        x = x.view(x.size(0), -1) # [B, 64]
        return self.classifier(x)
