import torch
import torch.nn as nn

class SSP(nn.Module):
    def __init__(self, pool_size = [1,2,4]):
        super().__init__()
        """
        Constructor of SSP(Spatial Pyramid Pooling)

        Args:
            pool_size -> Size of dynamic pooling
        """

        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(size) for size in pool_size
        ])

    def forward(self, X):
        features = []
        print(f'X in SSP {X.shape}')
        for pool in self.pools:
            pooled = pool(X)
            features.append(pooled.view(X.shape[0], - 1))
        return torch.cat(features, dim = 1)
