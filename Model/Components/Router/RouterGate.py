import torch.nn as nn
import torch.nn.functional as F

class RouterGate(nn.Module):
    def __init__(self, in_channel, hidden_size, num_experts):
        super().__init__()
        
        self.norm = nn.LayerNorm(in_channel)
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channel, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_experts)
        )

    def forward(self, X):
        # Global Average Pooling and Global Max Pooling
        gap = X.mean(dim = (2, 3)) # [B * P, C]
        gmp = X.max(dim = (2, 3)) # [B * P, C]
        X_cat = torch.cat([gap, gmp], dim = 1) # [B * P, 2 * C]

        # Layer Normalization
        X_norm = self.norm(X_cat)

        # MLP
        logits = self.mlp(X_norm)

        return logits

    