import torch
import torch.nn as nn

def conv3x3_same_block(in_features: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, kernel_size=(3,3), stride=(1,1), padding=1),
        nn.BatchNorm2d(out_features, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True)
    )

class NaiveCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        
        self.encoder = nn.Sequential(
            conv3x3_same_block(1, 16),
            conv3x3_same_block(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv3x3_same_block(32, 16),
            conv3x3_same_block(16, 8),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.classifier = nn.Sequential(
            # [-1, 16, 3, 15]
            nn.Linear(360, 360//2),
            nn.Linear(360//2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        flatten_features = self.flatten(features)
        logits = self.classifier(flatten_features)
        return logits