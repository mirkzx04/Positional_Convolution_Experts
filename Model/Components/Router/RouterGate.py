from numpy import dtype, indices
import torch
import torch.nn as nn
import torch.nn.functional as F

class RouterGate(nn.Module):
    def __init__(self, in_channel, hidden_size, num_experts, out_channel = 64):
        super().__init__()
        
        # self.projection = nn.Linear(in_channel, out_channel)
        self.norm = nn.LayerNorm(2 * in_channel)
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channel, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_experts)
        )
        # self.head_w = nn.Parameter(
        #     torch.empty(out_channel, num_experts)
        # )
        # nn.init.orthogonal_(self.head_w)

        # self.logit_scale = nn.Parameter(
        #     torch.tensor(float(10.0)).log()
        # )
        # self.bias = nn.Parameter(torch.zeros(num_experts))
        # self.initialize_weights()

    def forward(self, X):
        X = X.to(dtype=torch.float32) # [B*P, C, H, W]

        # Extract X statistics
        avarage_pooling = X.mean(dim = (2, 3)).to(dtype=torch.float32) # [B*P, C]
        max_pooling = X.amax(dim = (2, 3)).to(dtype=torch.float32 ) # [B*P, C]
        # min_pooling = X.amin(dim = (2, 3)).to(dtype = torch.float32) # [B*P, C]
        X_cat = torch.cat([avarage_pooling, max_pooling], dim = 1) # [B*P, 3 * C]
        
        # Norm and MLP
        X_norm = self.norm(X_cat).to(dtype=torch.float32)
        logits = self.mlp(X_norm).to(dtype=torch.float32)
        # logits_norm = F.normalize(logits, dim = -1)
        # w_n = F.normalize(self.head_w, dim = 0)
        # scale = self.logit_scale.exp().clamp(1.0, 100.0)
        # logits = (scale * (logits_norm @ w_n) + self.bias)

        return logits.to(dtype=torch.float32)

    def initialize_weights(self):
        for name, module in self.mlp.named_modules():
            if '0' in name:
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif '2' in name:
                nn.init.normal_(module.weight, mean = 0.0, std = 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

                
                actual_std = module.weight.std().item()
                actual_mean = module.weight.mean().item()

                print('Router gate')
                print('tgt = mean 0.0, std = 0.01')
                print(f'actual mean = {actual_mean:.6f}, actual std = {actual_std:.6f}')
    