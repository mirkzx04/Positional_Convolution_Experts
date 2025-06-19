import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans
from torch import nn

from .SSP import SSP

class Router(nn.Module):
    def __init__(self,num_experts, out_channel_key, enabled_ema=True, ema_alpha=0.99):
        super().__init__()
        """
        Router constructor

        Args:
            out_channel_key -> out channel for convolution of pixel projection
            num_experts -> number of experts in all layer, we use this number for compute cluster with K-Means
        """

        self.num_experts = num_experts

        self.ssp = SSP()

        self.ema_alpha = ema_alpha

        self.last_forward_cache = None
        self.cache_enabled = False

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

    def forward(self, patch, threshold, enable_ema=True):
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
        
        if not enable_ema and not isinstance(self.keys, nn.Parameter):
            # Clone keys in trainable parameter if not using EMA
            self.keys = nn.Parameter(self.keys.clone().detach(), requires_grad=True)

        # Compute cosine simlarity between patch embedding and keys
        logits = patch_emb @ self.keys.T

        # Calc softmax weights and applied threshold
        weights = F.softmax(logits, dim=-1)
        mask = weights > threshold
        weights_filtered = weights * mask.float()
        weights_filtered = weights_filtered / (weights_filtered.sum(dim=-1, keepdim=True) + 1e-8)

        if self.cache_enabled:
            self.last_forward_cache = {
                'patch_embeddings' : patch_emb.detach(),
                'cosine_similarities' : logits.detach(),
                'weights_raw' : weights.detach(),
                'weights_filtered' : weights_filtered,
                'mask' : mask.detach(),
                'threshold' : threshold,
                'input_shape' : patch.shape
            }

        if enable_ema:
            # Update keys with exponential moving average
            self.ema(patch_emb, weights)

        return weights_filtered
    
    def get_cached_metrics(self):
        """
        Get all metrics of last forward pass (if chace enabled)

        Returns:
            dict : All calculate metrics, None if cache disabled
        """

        if not self.cache_enabled or self.last_forward_cache is None:
            return None
        
        cache = self.last_forward_cache

        # Calculate metrics
        weights_filtered = cache['weights_filtered']
        weights = cache['weights_raw']

        # Expert assignments
        expert_assignments = torch.argmax(weights_filtered, dim = -1)
        assignment_confidence = torch.max(weights_filtered, dim = -1)[0]

        # Routing entropy
        weights_safe = weights_filtered + 1e-10
        routing_entropy = - torch.sum(weights_safe * torch.log(weights_safe), dim = -1)

        # Expert utilization
        num_patches = expert_assignments.numel()
        utilization = torch.zeros(self.num_experts, device = expert_assignments.device)
        for expert_idx in range(self.num_experts):
            utilization[expert_idx] = (expert_assignments == expert_idx).float().sum() / num_patches

        # Keys similarity matrix
        keys_norm = F.normalize(self.keys, dim=-1)
        keys_similarity = keys_norm @ keys_norm.T

        # Sparsity metrics
        sparsity_level = (weights_filtered == 0).float().mean()
        active_experts_per_patch = (weights_filtered > 0).float().sum(dim = -1)

        return {
            # Dati base (dal cache)
            'patch_embeddings': cache['patch_embeddings'],
            'cosine_similarities': cache['cosine_similarities'],
            'weights_raw': weights,
            'weights_filtered': weights_filtered,
            'threshold': cache['threshold'],
            
            # Metriche derivate
            'expert_assignments': expert_assignments,
            'assignment_confidence': assignment_confidence,
            'routing_entropy': routing_entropy,
            'mean_routing_entropy': routing_entropy.mean().item(),
            
            # Utilizzo esperti
            'expert_utilization': utilization,
            'max_expert_utilization': utilization.max().item(),
            'min_expert_utilization': utilization.min().item(),
            'utilization_std': utilization.std().item(),
            'utilization_entropy': -(utilization * torch.log(utilization + 1e-10)).sum().item(),
            
            # Keys similarity
            'keys_similarity_matrix': keys_similarity,
            'keys_max_similarity': keys_similarity[~torch.eye(self.num_experts, dtype=bool, device=keys_similarity.device)].max().item(),
            'keys_min_similarity': keys_similarity[~torch.eye(self.num_experts, dtype=bool, device=keys_similarity.device)].min().item(),
            'keys_mean_similarity': keys_similarity[~torch.eye(self.num_experts, dtype=bool, device=keys_similarity.device)].mean().item(),
            
            # Sparsity e distribuzione
            'sparsity_level': sparsity_level.item(),
            'active_experts_per_patch_mean': active_experts_per_patch.mean().item(),
            'active_experts_per_patch_std': active_experts_per_patch.std().item(),
            'mean_max_weight_raw': weights.max(dim=-1)[0].mean().item(),
            'mean_max_weight_filtered': weights_filtered.max(dim=-1)[0].mean().item(),
            'low_confidence_patches_pct': (assignment_confidence < 0.5).float().mean().item() * 100
        }

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
            centroids = kmeans.cluster_centers_

            # Normalize centroids
            keys = torch.tensor(centroids, dtype=torch.float32)
            self.keys = F.normalize(keys, dim=-1)

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