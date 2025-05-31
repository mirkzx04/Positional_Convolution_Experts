import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans
from torch import nn

class Router(nn.Module):
    def __init__(self,num_experts):
        super().__init__()
        """
        Router constructor
        """

        self.num_experts = num_experts

    def initialize_keys(self, patches):
        """
        Initialize key for routing throught K-Means

        Args : 
            patches -> Tensor (B, number_patch, C + 2, H, W)
                    where + 2 is positional information and H = W = patch_size
        """
        # Reshape patches from (B, P, C + 2, H, W) to (BxP, (C + 2)xCxHxW)
        B, P, C, H, W = patches.shape
        D = C*H*W
        patches_flat = patches.view(B*P, D)

        patches_flat = patches_flat.cpu().numpy()

        # Initialize KMeans and fit for get centroids
        kmeans = KMeans(n_clusters = self.num_experts, n_init = 'auto', random_state = 42)
        kmeans.fit(patches_flat)
        centroids = kmeans.cluster_centers_

        # Normalize centroids
        keys = torch.tensor(centroids, dtype=torch.float32)
        keys = F.normalize(keys, dim=-1)

        return keys