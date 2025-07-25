import math

import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

from torch import nn

from .PatchEmbedder import PatchEmbedder

class Router(nn.Module):
    def __init__(self,num_experts, num_layers, embed_dim, pce_layer_info, ema_alpha=0.9):
        super().__init__()
        """
        Router constructor

        Args:
            out_channel_key -> out channel for convolution of pixel projection
            num_experts -> number of experts in all layer, we use this number for compute cluster with K-Means
            num_layers -> number of layers in PCE Network
            proj_channel -> list of out channels for projection convolution in each layer
            ema_alpha -> Exponential moving average alpha for update keys
        """

        self.num_experts = num_experts
        self.ema_alpha = ema_alpha

        self.layers_cache = []
        self.cache_enabled = True

        self.embedding_dim = embed_dim 
        
        self.embedders = nn.ModuleList([
            PatchEmbedder(info['embd_channel'], info['patch_size'], info['embedd_dim']) 
            for info in pce_layer_info
        ])

        # Create keys
        self.keys = nn.ParameterList([
            nn.Parameter(torch.randn(num_experts, info['embedd_dim'], dtype=torch.float32), requires_grad=True)
            for info in pce_layer_info
        ])

        # Set parameters of adaptive threshold
        self.logit_temp = nn.Parameter(torch.tensor(5.0, dtype=torch.float32), requires_grad=True)
        self.mask_beta = nn.Parameter(torch.tensor(10., dtype=torch.float32), requires_grad=True)
        self.min_experts_active = max(1, 
            num_experts // 2 if num_experts <= 4 
            else (2 if num_experts < 8 else num_experts // 4)
        )

        self.wth_max_c = nn.Parameter(torch.tensor(0.4, dtype=torch.float32), requires_grad=True) 
        self.wth_entropy_c = nn.Parameter(torch.tensor(0.3, dtype=torch.float32), requires_grad=True)
        self.wth_gap_c = nn.Parameter(torch.tensor(0.3, dtype=torch.float32), requires_grad=True)

    def set_keys_trainable(self, trainable = False):
        if trainable:
            for k in self.keys:
                k.requires_grad = True

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

    def forward(self, patch, layer_idx, threshold, hard_threshold = False):
        """
        Forward method of Router

        Args : 
            patch -> tensor (B x nP, C + 4, nH, nW)
            threshold -> float, threshold for experts scores
            enable_ema -> bool, enable or disable exponential moving average for keys
            layer_idx -> int, index of current layer
        
        Returns:
            weights -> Tensor (B x nP, num_experts)
            where B is batch size, nP is number of patches, num_experts is number of experts in layer
        """
        emb = self.embedders[layer_idx](patch)
        key = F.normalize(self.keys[layer_idx], dim = -1)
        # Compute cosine simlarity between patch embedding and keys
        logit = emb @ key.T
        logit /= (emb.shape[-1])**0.5
        # cosine_scaled = cosine * self.logit_temp

        # Calc softmax weights and applied threshold
        weights = F.softmax(logit * self.logit_temp, dim=-1)
        
        # Compute adaptive threshold
        # adaptive_threshold = self.compute_adaptive_threshold(weights=weights, base_threshold=threshold)

        # if hard_threshold == False:
        #     # Soft threshdol
        #     soft_mask = torch.sigmoid(
        #         (self.mask_beta + 1e-8) * (weights - adaptive_threshold)
        #     )

        #     weights_filtered = weights * soft_mask
        # else:
        #     # Hard threshold
        #     hard_mask = (weights > adaptive_threshold).float()
        #     weights_filtered = weights * hard_mask

        # Normalize weights
        # sum_weights = weights.sum(dim=-1, keepdim=True)
        # weights = weights / torch.clamp(sum_weights, min = 1e-8)

        # Save cache
        if self.cache_enabled:
            self.layers_cache.append({
                'embedding' : emb,
                'logits' : logit,
                'weights' : weights,
                'keys' : key
            })

        if self.keys[layer_idx].requires_grad == False:
            # Update keys with exponential moving average
            self.ema(emb, weights, layer_idx)

        return weights

    def get_cached_metrics(self):
        if self.cache_enabled:
            return self.last_forward_cache
        
        return None

    def initialize_keys(self, patches, layer_idx):
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
            base = torch.empty(self.embedding_dim, self.embedding_dim)

            nn.init.orthogonal_(base)
            key = base[:4]
            key += 0.05 * torch.rand_like(key)
            self.keys[layer_idx].data.copy_(key)

    def ema(self, patch_embedding, weights, layer_idx):
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

            # Update all keys 
            self.keys[layer_idx].data.mul_(self.ema_alpha).add_(
                centroids * (1.0 - self.ema_alpha)
            )

    def enable_cache(self):
        self.cache_enabled = True

    def disable_cache(self):
        self.cache_enabled = False
        self.layers_cache = None

    def clear_cache(self):
        self.layers_cache.clear()

    def get_cache(self):
        return self.layers_cache