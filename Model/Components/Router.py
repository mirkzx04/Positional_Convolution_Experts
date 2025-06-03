import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans
from torch import nn

from .SSP import SSP

class Router(nn.Module):
    def __init__(self,num_experts, out_channel_key):
        super().__init__()
        """
        Router constructor

        Args:
            out_channel_key -> out channel for convolution of pixel projection
            num_experts -> number of experts in all layer, we use this number for compute cluster with K-Means
        """

        self.num_experts = num_experts

        self.ssp = SSP()

    def forward(self, patch):
        """
        Forward method of Router

        Args : 
            patch -> tensor (B x nP, C + 4, nH, nW)
        """
        patch_emb = self.ssp(patch)
        patch_emb = F.normalize(patch_emb, dim=-1)
        
        # Compute cosine simlarity between patch embedding and keys
        logits = patch_emb @ self.keys.T
        
        weights = F.softmax(logits, dim=-1)
        return weights

    def initialize_keys(self, patches):
        """
        Initialize key for routing throught K-Means

        Args : 
            patches -> Tensor (B, number_patch, C + 2, H, W)
                    where C + 2 is positional information and H = W = patch_size
        """

        with torch.no_grad():
            # Reshape patch (B, nP, C + 4, nH, nW) -> (B x nP, C + 4, nH, nW) and applied projection convolution
            # pixel projection for create patch embedding with SSP
            reshape_patches = self.reshape_patch(patch=patches)
            conv_proj = nn.Conv2d(in_channels=7, kernel_size=3, out_channels=8)
            patch_proj = conv_proj(reshape_patches)

            # Applied SSP for get patch embedding using K-Means
            patch_emb = self.ssp(patch_proj)

            # Initialize KMeans and fit for get centroids
            kmeans = KMeans(n_clusters = self.num_experts, n_init = 'auto', random_state = 42)
            kmeans.fit(patch_emb.detach().cpu().numpy())
            centroids = kmeans.cluster_centers_

            # Normalize centroids
            keys = torch.tensor(centroids, dtype=torch.float32)
            self.keys = F.normalize(keys, dim=-1)

    def reshape_patch(self, patch):
        # Reshape patches from (B, P, C + 2, H, W) to (BxP, (C + 2), H, W)
        B, P, C, H, W = patch.shape
        return patch.reshape(B*P, C, H, W)