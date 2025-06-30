import math

import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

from torch import nn

from .SSP import SSP

class Router(nn.Module):
    def __init__(self,num_experts, ema_alpha=0.99):
        super().__init__()
        """
        Router constructor

        Args:
            out_channel_key -> out channel for convolution of pixel projection
            num_experts -> number of experts in all layer, we use this number for compute cluster with K-Means
        """

        self.num_experts = num_experts
        self.ema_alpha = ema_alpha

        self.last_forward_cache = None
        self.cache_enabled = False

        # Get SSP Classs, use for embedding
        self.ssp = SSP()

        # Create keys
        self.keys = nn.Parameter(torch.empty(0), requires_grad=False)

        # Set parameters of adaptive threshold
        self.logit_temp = nn.Parameter(torch.tensor(5.)) 
        self.mask_beta = nn.Parameter(torch.tensor(10.))
        self.min_experts_active = max(1, 
            num_experts // 2 if num_experts <= 4 
            else (2 if num_experts < 8 else num_experts // 4)
        )

        self.wth_max_c, self.wth_entropy_c, self.wth_gap_c = 0.4, 0.3, 0.3

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

    def compute_gap(self, sort_weights):
        """
        Compute gap between top-1 expert and successive experts 

        Args : 
            sort_weights : torch.tensor with shape [B xnP, num_experts] - sorted softmax weights 
        """
        k = min(4, self.num_experts - 1)
        mean_rest = sort_weights[:, 1:k+1].mean(dim = -1, keepdim = True)
        gap = (sort_weights[:, 0:1] - mean_rest) / (sort_weights[:, 0:1] + 1e-8)

        return torch.clamp(gap, 0., 1.)
    
    def compute_adaptive_threshold(self, weights, base_threshold):  
        """
        Compute adaptive threshold that it based of multiple statistics metrics

        Args:
            weights : Tensor(B x nP, num_experts) - softmax weights
            base_threshold : Float
        """
        # Sorted weights
        sort_weights, _ = weights.sort(dim = -1, descending = True)

        # 1 - w_max => if w_max is high than threshold is lower, w_max is lower than threshold is high
        max_component = 1.0 - sort_weights[:, 0:1]
        mean_weight, std_weight = weights.mean(dim = -1, keepdim = True), weights.std(dim = -1, keepdim = True)

        # Compute entropy, high entropy (uniform distribution) => lower threshold
        # lower entropy (concetrated distribution) => high threshold
        entropy = -(weights * torch.log(weights + 1e-18)).sum(dim = -1, keepdim = True)
        max_entropy = math.log(self.num_experts)
        entropy_component = 1.0 - entropy / max_entropy
        
        # Compute top-1 gap
        gap_component = self.compute_gap(sort_weights)

        # Linear combination of the top router expert weight,
        # entropy and gap
        # all weights of components is a learnable parameters to make sure that the model
        # can understand the importance of each components
        adaptive_factor = (
            self.wth_max_c * max_component + self.wth_entropy_c * entropy_component + self.wth_gap_c * gap_component
        )

        # Compute adaptive threshold
        adaptive_threshold = base_threshold * (0.5 + adaptive_factor)

        # Defines min and max threshold
        min_threshold = torch.maximum(
            torch.tensor(0.05, device=weights.device),
            mean_weight - 0.5 * std_weight
        )
        max_threshold = torch.minimum(
            torch.tensor(0.7, device=weights.device),
            sort_weights[:, 0:1] - 0.1 * std_weight
        )
        
        adaptive_threshold = torch.clamp(adaptive_threshold, min=min_threshold, max=max_threshold)
        
        if self.min_experts_active <= self.num_experts:
            kth_weight = sort_weights[:, self.min_experts_active-1 : self.min_experts_active]
            adaptive_threshold = torch.minimum(adaptive_threshold, kth_weight * 0.9)
        
        return adaptive_threshold

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
        scaled_logits = logits / self.logit_temp + 1e-8

        # Calc softmax weights and applied threshold
        weights = F.softmax(scaled_logits, dim=-1)
        
        # Compute adaptive threshold
        adaptive_threshold = self.compute_adaptive_threshold(weights=weights, base_threshold=threshold)

        if hard_threshold == False:
            # Soft threshdol
            soft_mask = torch.sigmoid(
                (self.mask_beta + 1e-8) * (weights - adaptive_threshold)
            )

            weights_filtered = weights * soft_mask
        else:
            # Hard threshold
            hard_mask = (weights > adaptive_threshold).float()
            weights_filtered = weights * hard_mask

        # Normalize weights
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
                'max_weights': weights.max(dim = -1, keepdim = True)[0].detach(),
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
            # Transpose weights from [nxP, E] -> [E, n x P]
            w_t = weights.t()

            # Sum weights for each experts and calc centroids
            denom = w_t.sum(dim=1, keepdim = True) + 1e-8
            centroids = (w_t @ patch_embedding) / denom

            # Normalize centroids
            centroids = F.normalize(centroids, dim = -1)

            # Update all keys 
            self.keys.data.mul_(self.ema_alpha).add_(
                centroids * (1.0 - self.ema_alpha)
            )

            self.keys.data = F.normalize(self.keys.data, dim = -1)

    def reshape_patch(self, patch):
        # Reshape patches from (B, P, C + 2, H, W) to (BxP, (C + 2), H, W)
        B, P, C, H, W = patch.shape
        return patch.reshape(B*P, C, H, W)