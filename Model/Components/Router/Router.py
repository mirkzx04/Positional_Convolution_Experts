import math

import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

from torch import nn

from .RouterGate import RouterGate

class Router(nn.Module):
    def __init__(self,num_experts, num_layers, pce_layer_info):
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

        self.layers_cache = []
        self.cache_enabled = True
        
        self.gates = nn.ModuleList([
            RouterGate(info[0], info[1], num_experts) 
            for info in pce_layer_info
        ])

        # Set parameters of adaptive threshold
        self.mask_beta = nn.Parameter(torch.tensor(10., dtype=torch.float32), requires_grad=True)
        self.min_experts_active = max(1, 
            num_experts // 2 if num_experts <= 4 
            else (2 if num_experts < 8 else num_experts // 4)
        )

        self.wth_max_c = nn.Parameter(torch.tensor(0.4, dtype=torch.float32), requires_grad=True) 
        self.wth_entropy_c = nn.Parameter(torch.tensor(0.3, dtype=torch.float32), requires_grad=True)
        self.wth_gap_c = nn.Parameter(torch.tensor(0.3, dtype=torch.float32), requires_grad=True)

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

    def forward(self, patch, layer_idx, threshold, current_epoch=None):
        """
        Forward method of Router

        Args : 
            patch -> tensor (B x nP, C + 4, nH, nW)
            threshold -> float, threshold for experts scores
            current_epoch -> int, current training epoch, used to determine routing strategy
            layer_idx -> int, index of current layer
        
        Returns:
            weights -> Tensor (B x nP, num_experts)
            where B is batch size, nP is number of patches, num_experts is number of experts in layer
        """
        logits = self.gates[layer_idx](patch)

        # Compute softmax weights
        weights = F.softmax(logits, dim=-1) 

        # Compute adaptive threshold
        adaptive_threshold = self.compute_adaptive_threshold(weights=weights, base_threshold=threshold)

        if current_epoch is not None and current_epoch < 20:
            # Create uniform weights when epoch < 20 or epoch not provided
            # Shape: [B, P, num_experts] with uniform weights
            weights_norm = torch.ones_like(weights) / self.num_experts
        elif current_epoch is not None and 20 <= current_epoch <= 25:
            # Create soft weights when 20 <= epoch <= 25
            tau = max(0.1, 1.0 -( current_epoch - 20) * 0.18)
            weights_norm = torch.sigmoid((weights - adaptive_threshold) / tau)

            weights_norm = weights_norm / torch.clamp(weights_norm.sum(dim=-1, keepdim=True), min = 1e-8)
        elif current_epoch > 25 or self.training == False:
            # Create hard weights when epoch > 25
            tau = 0.1
            hard_mask = (weights > adaptive_threshold).float()
            soft_mask = torch.sigmoid((weights - adaptive_threshold) / tau)
            filtered_weights = hard_mask + (soft_mask - hard_mask).detach()

            # Normalize weights
            weights_norm = filtered_weights / torch.clamp(filtered_weights.sum(dim=-1, keepdim=True), min = 1e-8)
        
        # Save cache
        if self.cache_enabled:
            self.layers_cache.append({
                'logits' : logits,
                'weights' : weights,
                'norm_weights' : weights_norm,
                'base_threshold' : threshold,
                'adaptive_threshold' : adaptive_threshold
            })

        return weights_norm

    def get_cached_metrics(self):
        if self.cache_enabled:
            return self.last_forward_cache
        
        return None
        
    def enable_cache(self):
        self.cache_enabled = True

    def disable_cache(self):
        self.cache_enabled = False
        self.layers_cache = None

    def clear_cache(self):
        self.layers_cache.clear()

    def get_cache(self):
        return self.layers_cache