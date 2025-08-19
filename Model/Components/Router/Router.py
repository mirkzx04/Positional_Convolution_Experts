import math

import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

from torch import nn

from .RouterGate import RouterGate

class Router(nn.Module):
    def __init__(self,num_experts, num_layers, pce_layer_info, nucleus_sampling_p):
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

        self.p = nucleus_sampling_p
        self.min_k = 1

        self.val_tau = 0.9
        self.val_p = 0.8
        self.val_min_k = 1

    # def compute_gap(self, sort_weights):
    #     """
    #     Compute gap between top-1 expert and successive experts 

    #     Args : 
    #         sort_weights : torch.tensor with shape [B xnP, num_experts] - sorted softmax weights 
    #     """
    #     k = min(4, self.num_experts - 1)
    #     mean_rest = sort_weights[:, 1:k+1].mean(dim = -1, keepdim = True)
    #     gap = (sort_weights[:, 0:1] - mean_rest) / (sort_weights[:, 0:1] + 1e-8)

    #     return torch.clamp(gap, 0., 1.)
    
    # def compute_adaptive_threshold(self, weights, base_threshold):  
    #     """
    #     Compute adaptive threshold that it based of multiple statistics metrics

    #     Args:
    #         weights : Tensor(B x nP, num_experts) - softmax weights
    #         base_threshold : Float
    #     """
    #     # Sorted weights
    #     sort_weights, _ = weights.sort(dim = -1, descending = True)

    #     # 1 - w_max => if w_max is high than threshold is lower, w_max is lower than threshold is high
    #     max_component = 1.0 - sort_weights[:, 0:1]
    #     mean_weight, std_weight = weights.mean(dim = -1, keepdim = True), weights.std(dim = -1, keepdim = True)

    #     # Compute entropy, high entropy (uniform distribution) => lower threshold
    #     # lower entropy (concetrated distribution) => high threshold
    #     entropy = -(weights * torch.log(weights + 1e-18)).sum(dim = -1, keepdim = True)
    #     max_entropy = math.log(self.num_experts)
    #     entropy_component = 1.0 - entropy / max_entropy
        
    #     # Compute top-1 gap
    #     gap_component = self.compute_gap(sort_weights)

    #     # Linear combination of the top router expert weight,
    #     # entropy and gap
    #     # all weights of components is a learnable parameters to make sure that the model
    #     # can understand the importance of each components
    #     adaptive_factor = (
    #         self.wth_max_c * max_component + self.wth_entropy_c * entropy_component + self.wth_gap_c * gap_component
    #     )

    #     # Compute adaptive threshold
    #     adaptive_threshold = base_threshold * (0.5 + adaptive_factor)

    #     # Defines min and max threshold
    #     min_threshold = torch.maximum(
    #         torch.tensor(0.05, device=weights.device),
    #         mean_weight - 0.5 * std_weight
    #     )
    #     max_threshold = torch.minimum(
    #         torch.tensor(0.7, device=weights.device),
    #         sort_weights[:, 0:1] - 0.1 * std_weight
    #     )
        
    #     adaptive_threshold = torch.clamp(adaptive_threshold, min=min_threshold, max=max_threshold)
        
    #     if self.min_experts_active <= self.num_experts:
    #         kth_weight = sort_weights[:, self.min_experts_active-1 : self.min_experts_active]
    #         adaptive_threshold = torch.minimum(adaptive_threshold, kth_weight * 0.9)
        
    #     return adaptive_threshold

    def top_p_mask(self, weights, p):
        """
        Create top-p mask for routing
        """
        # Sort weights
        w_sorted, idx = torch.sort(weights, descending = True, dim = -1)

        # Compute cumulative sum
        cumsum = torch.cumsum(w_sorted, dim = -1)
        rank = torch.arange(w_sorted.size(-1), device = weights.device).view(*((1,)*(weights.ndim - 1)), -1)

        keep_sorted = (cumsum <= p) | (rank < self.min_k)
        mask = torch.zeros_like(weights)
        mask.scatter_(-1, idx, keep_sorted.type_as(weights))

        return mask

    def tau_scheduler(self, current_epoch):
        """
        Scheduler for tau
        """
        if current_epoch is not None and current_epoch < 30:
            return 1.5
        elif current_epoch is not None and 30 <= current_epoch <= 70:
            t0, t1, T = 1.5, 0.85, 80
            e = min(max(current_epoch, 0), T)
            cos = 0.5 * (1 + torch.cos(torch.tensor(e/T * math.pi)))
            return (t1 + (t0 - t1) * cos.item())
        elif current_epoch > 70:
            return 0.75
    
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

        
        if self.training:
            # Compute router temperature
            tau = self.tau_scheduler(current_epoch)
            if tau > 0:
                weights = F.softmax(logits / tau, dim=-1)
            else:
                weights = F.softmax(logits, dim=-1)
            
            # Compute top-p mask
            mask = self.top_p_mask(weights, p = self.p)

            if current_epoch < 30:
                # Create uniform weights with blending factor
                alpha = 1.0 if current_epoch is None else float(min(1.0, current_epoch / 30))
                uniform = weights.new_full(weights.shape, 1.0 / self.num_experts)

                weights_norm = (1 - alpha) * uniform + alpha * weights
                
            elif 30 <= current_epoch <= 70:
                h = (current_epoch - 30) / 40
                h = float(max(0.0, min(1.0, h)))

                # Compute hard weights
                masked_weights = weights * mask
                denom = masked_weights.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                hard_weights = masked_weights / denom

                # apply straight through estimator
                weights_norm = hard_weights.detach() + (weights - weights.detach())

                weights_norm = (1.0- h)*weights + h * weights_norm

            elif current_epoch > 70:
                # Apply top-p mask to weights
                masked_weights = weights * mask

                # Calculate denominator for normalization, with minimum value of 1e-10
                denom = masked_weights.sum(dim=-1, keepdim=True).clamp_min(1e-10)

                # Normalize masked weights
                fw = masked_weights / denom

                # Apply straight-through estimator for gradient
                weights_norm = fw.detach() + weights - weights.detach()
        else:
            weights, weights_norm, mask = self.validation_mode(logits)
            tau = self.val_tau

        # Save cache
        if self.cache_enabled:
            self.layers_cache.append({
                'logits' : logits,
                'weights' : weights,
                'norm_weights' : weights_norm,
                'tau' : tau,
                'mask' : mask
            })

        return weights_norm

    def validation_mode(self, logits):
        """
        Validation mode for router
        """
        probs = F.softmax(logits / self.val_tau, dim=-1)

        # Compute top-p mask
        mask = self.top_p_mask(probs, p = self.val_p)
        masked = probs * mask

        denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        weights_norm = masked / denom

        return probs, weights_norm, mask

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