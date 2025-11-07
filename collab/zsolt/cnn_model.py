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

#Regularized version of SmallCNN for AudioMNIST digit classification.

# What’s new:
#- Added dropout layers between convolution blocks and before the final classifier
#  to reduce overfitting and stabilize validation accuracy.
#- Still uses AdaptiveAvgPool2d so variable-length spectrograms work seamlessly.

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()

        # === Feature extraction layers ===
        self.features = nn.Sequential(
            # Block 1: learns basic frequency shapes (edges / harmonics)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),     # halves mel/time dimensions
            nn.Dropout2d(0.10),  # disables 10 % of feature maps randomly

            # Block 2: captures more complex temporal patterns
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            # Block 3: high-level abstractions (speaker-independent cues)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Global pooling removes dependency on time length
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # === Classifier head ===
        self.classifier = nn.Sequential(
            nn.Dropout(0.30),        # further regularization before dense layer
            nn.Linear(64, n_classes) # final logits (10 digits)
        )

    def forward(self, x):
        # Extract convolutional features
        x = self.features(x)         # [B, 64, 1, 1]
        # Flatten pooled output to vector
        x = x.view(x.size(0), -1)    # [B, 64]
        # Classify into logits
        return self.classifier(x)
