import torch.nn as nn
import torch.nn.functional as F

class RouterGate(nn.Module):
    def __init__(self, in_channel, patch_size, num_experts, pooling_size=4):
        super().__init__()
        
        self.flatten = nn.Flatten()
        flatten_size = in_channel * patch_size * patch_size

        self.mlp = nn.Sequential(
            nn.Linear(flatten_size, embed_dim)
        )


    def forward(self, X):
        X_flat = self.flatten(X)
        logits = self.mlp(X_flat)

        return logits

    