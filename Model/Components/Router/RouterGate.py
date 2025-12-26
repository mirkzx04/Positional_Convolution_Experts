from numpy import dtype, indices
import torch
import torch.nn as nn
import torch.nn.functional as F

class RouterGate(nn.Module):
    def __init__(self, in_channel, num_experts):
        super().__init__()
        
        # self.projection = nn.Linear(in_channel, out_channel)
        in_size = 2 * in_channel
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_size),
            nn.Linear(in_size, num_experts, bias=True),
        )
        self.k = 1
        self.initialize_weights()

    def forward(self, X):
        X = X.to(dtype=torch.float32) # [B*P, C, H, W]

        # Extract X statistics
        avarage_pooling = X.mean(dim = (2, 3)).to(dtype=torch.float32) # [B*P, C]
        max_pooling = X.amax(dim = (2, 3)).to(dtype=torch.float32 ) # [B*P, C]
        # min_pooling = X.amin(dim = (2, 3)).to(dtype = torch.float32) # [B*P, C]
        X_cat = torch.cat([avarage_pooling, max_pooling], dim = 1) # [B*P, 3 * C]
        
        # Norm and MLP
        logits = self.mlp(X_cat).to(dtype=torch.float32)
        logits = logits - self.k * logits.mean(dim = -1, keepdim = True)

        return logits.to(dtype=torch.float32)

    def initialize_weights(self):
        for _, module in self.mlp.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    