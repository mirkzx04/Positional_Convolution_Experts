import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedder(nn.Module):
    def __init__(self, in_channel, patch_size, embed_dim = 400, pooling_size=4):
        super().__init__()
        
        self.flatten = nn.Flatten()
        flatten_size = in_channel * patch_size * patch_size

        self.mlp = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )

        self.fc_skip = nn.Linear(flatten_size, embed_dim)
        self.alpha = 0.1

    def forward(self, X):
        X_flat = self.flatten(X)
        emb = self.mlp(X_flat)
        skip_emb = self.fc_skip(X_flat)
        content_emb = (emb + self.alpha * skip_emb)

        return content_emb

    