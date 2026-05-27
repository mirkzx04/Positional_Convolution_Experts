from numpy import dtype, indices
import torch
import torch.nn as nn
import torch.nn.functional as F

class RouterGate(nn.Module):
    def __init__(self, in_channel, num_experts, patch_h, patch_w):
        super().__init__()
        
        # Initialize router patch representation 
        self.expert_emb = nn.Parameter(
            torch.randn(num_experts, in_channel, patch_h, patch_w) * 0.02
        ) # Shape : [E, C_in, H_p, W_p]

        self.initialize_weights()

    def forward(self, X, positional_features):
        X = X.to(dtype=torch.float32) # [B*P, C, H, W]
        Xn = F.layer_norm(X, X.shape[1:])

        c_x = Xn.shape[1]
        x_w = self.expert_emb[:, :c_x].to(dtype=Xn.dtype)

        # Compute logits without positional features 
        logits = torch.einsum("nchw,echw->ne",Xn , x_w) # Shape : [N, E]

        # Adding positional features
        positional_features = positional_features.flatten(1).float()
        pos_w = self.expert_emb[:, c_x:].sum(dim=(-1, -2)).to(dtype=positional_features.dtype)
        logits = logits + torch.einsum("nf,ef->ne", positional_features, pos_w)

        logits = logits.float()

        return logits.to(dtype=torch.float32)

    def initialize_weights(self):
        # expert_emb is used as a linear projection from a flattened patch
        # [C, H, W] -> [E], so initialize it like a Linear(C*H*W, E) weight.
        nn.init.xavier_uniform_(self.expert_emb.flatten(1))
    
