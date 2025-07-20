import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedder(nn.Module):
    def __init__(self, in_channel, patch_size, embed_dim = 400, pooling_size=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(patch_size // 2)
        )
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(128 * patch_size // 2 * patch_size // 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim)
        )

    def forward(self, X):
        X = self.cnn(X)
        X = self.flatten(X)
        X = self.mlp(X)
        X = F.normalize(X)
        return X

    