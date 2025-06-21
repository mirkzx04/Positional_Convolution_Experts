import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans
from torch import nn

from .SSP import SSP

class Router(nn.Module):
    def __init__(self,num_experts, temperature, ema_alpha=0.99):
        super().__init__()
        """
        Router constructor

        Args:
            out_channel_key -> out channel for convolution of pixel projection
            num_experts -> number of experts in all layer, we use this number for compute cluster with K-Means
        """

        self.num_experts = num_experts
        self.temperature = temperature
        self.ema_alpha = ema_alpha

        self.last_forward_cache = None
        self.cache_enabled = False

        # Get SSP Classs, use for embedding
        self.ssp = SSP()

        # Create keys
        self.keys = nn.Parameter(torch.empty(0), requires_grad=False)

    def enable_metrics_cache(self):
        """
        Enable caching of the metrics during forward
        """
        self.cache_enabled = True

    def disable_metrics_cache(self):
        """
        Disable caching of the metrics for training
        """ 

        self.cache_enabled = False
        self.last_forward_cache = None

    def set_keys_trainable(self, trainable = False):
        if trainable:
            self.keys.requires_grad_(True)

    def forward(self, patch, threshold, hard_threshold = False):
        """
        Forward method of Router

        Args : 
            patch -> tensor (B x nP, C + 4, nH, nW)
            threshold -> float, threshold for experts scores
            enable_ema -> bool, enable or disable exponential moving average for keys
        
        Returns:
            weights -> Tensor (B x nP, num_experts)
            where B is batch size, nP is number of patches, num_experts is number of experts in layer
        """
        patch_emb = self.ssp(patch)
        patch_emb = F.normalize(patch_emb, dim=-1)

        # Compute cosine simlarity between patch embedding and keys
        logits = patch_emb @ self.keys.T

        # Calc softmax weights and applied threshold
        weights = F.softmax(logits, dim=-1)
        
        # Adaptive threshold
        max_weight = weights.max(dim=-1, keepdim=True)[0]
        adaptive_threshold = torch.clamp(
            threshold * (2.0 - max_weight),
            min = 0.01, max = 0.8
        )

        if hard_threshold == False:
            # Soft threshdol
            soft_mask = torch.sigmoid(
                self.temperature * (weights - adaptive_threshold)
            )

            weights_filtered = weights * soft_mask
        else:
            # Hard threshold
            hard_mask = (weights > adaptive_threshold).float()
            weights_filtered = weights * hard_mask

        sum_weights = weights_filtered.sum(dim=-1, keepdim=True)
        weights_filtered = weights_filtered / torch.clamp(sum_weights, min = 1e-8)

        if self.cache_enabled:
            self.last_forward_cache = {
                'patch_embeddings': patch_emb.detach(),
                'cosine_similarities': logits.detach(),
                'weights_raw': weights.detach(),
                'weights_filtered': weights_filtered.detach(),
                'base_threshold': threshold,  
                'adaptive_threshold': adaptive_threshold.detach(),
                'max_weights': max_weight.detach(), 
                'hard_threshold_used': hard_threshold
            }

        if not self.keys.requires_grad:
            # Update keys with exponential moving average
            self.ema(patch_emb, weights_filtered)

        return weights_filtered

    def get_cached_metrics(self):
        if self.cache_enabled:
            return self.last_forward_cache
        
        return None

    def initialize_keys(self, patches):
        """
        Initialize key for routing throught K-Means

        Args : 
            patches -> Tensor (B, number_patch, C + 2, H, W)
                    where C + 2 is positional information and H = W = patch_size
        Returns:
            None, but initialize keys as nn.Parameter with shape (num_experts, C + 4)
            where C + 4 is number of channels in patch embedding (C + 2 positional information + 2 pixel coordinates)
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
            centroids = torch.tensor(kmeans.cluster_centers_, device=patch_emb.device)
            centroids = F.normalize(centroids, dim = -1)

            self.keys.data = centroids

    def ema(self, patch_embedding, weights):
        """
        Exponential moving average for update keys

        Args:
            patch_embedding -> Tensor (B x nP, C + 4)
            weights -> Tensor (B x nP, num_experts)
                where B is batch size, nP is number of patches, num_experts is number of experts in layer
        Returns:
            None, but update keys in place
        """
        with torch.no_grad():
            for expert_idx in range(self.num_experts):
                expert_weights = weights[:, expert_idx]

                # Only update is some patches are assigned to this expert
                if expert_weights.sum() > 0:
                    # Compute weighted centroid of patches assigned to this expert
                    weighted_patches = patch_embedding * expert_weights.unsqueeze(-1)
                    weighted_centroid = weighted_patches.sum(dim=0) / (expert_weights.sum() + 1e-8)

                    # Normalize the centroid
                    weighted_centroid = F.normalize(weighted_centroid, dim=-1)

                    # Update the EMA keys for this expert
                    self.keys.data[expert_idx] = (
                        self.ema_alpha * self.keys.data[expert_idx] +
                        (1 - self.ema_alpha) * weighted_centroid
                    )

                    self.keys.data[expert_idx] = F.normalize(self.keys.data[expert_idx], dim=-1)

    def reshape_patch(self, patch):
        # Reshape patches from (B, P, C + 2, H, W) to (BxP, (C + 2), H, W)
        B, P, C, H, W = patch.shape
        return patch.reshape(B*P, C, H, W)