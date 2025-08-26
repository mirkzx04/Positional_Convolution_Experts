import math

import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

from torch import nn

# from .RouterGate import RouterGate

class Router(nn.Module):
    def __init__(self,num_experts, num_layers, noise_epsilon = 1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.num_layers = num_layers

        self.noise_epsilon = noise_epsilon

    def forward(self, X, router_gate, current_epoch = None):
        """
        Router forward method

        Args:
            X (torch.Tensor, shape [B*P, C, H, W]) -> Input
            router_gate (torch.Tensor, shape [B*P, C, num_experts]) -> Router gate
        """

        N, C, H, W = X.shape

        logits = router_gate(X) # [N, num_experts]

        # Add noise to logits in training mode
        if self.training:
            noise = logits.new_empty(logits.shape).uniform_(1 - self.noise_epsilon, 1 + self.noise_epsilon)
            logits = logits + noise

        # Compute z loss
        z_loss = self.z_loss(logits)

        if current_epoch is None or not self.training or (current_epoch is not None and current_epoch >= 30):
            # Apply softmax to get probabilities
            probs = F.softmax(logits.float(), dim = 1) # [N, num_experts]

            # Top-1 : Index of best expert 
            expert_gate, expert_index = probs.max(dim = 1) # [N]
            expert_mask = F.one_hot(expert_index, num_classes = self.num_experts) # [N, num_experts]

            # Compute aux loss
            aux_loss = self.aux_loss(expert_mask, probs)

            # Expert capacity
            if self.training:
                capcity_factor = getattr(self, 'capcity_factor', 1.25)
            else : 
                capcity_factor = 1.50
            
            Ccap = max(1, math.ceil(capcity_factor * N / self.num_experts))

            # List of patches per expert
            position_in_exp = torch.cumsum(expert_mask, dim = 0) - 1.0 # [N, num_experts]
            within_capacity = (position_in_exp < Ccap).to(probs.dtype) # [N, num_experts]
            expert_mask = expert_mask.to(probs.dtype) * within_capacity # [N, num_experts]

            expert_mask_flat = expert_mask.sum(dim = -1) # [N]
            expert_gate = expert_gate * expert_mask_flat

            pos_idx = position_in_exp.clamp(min = 0, max = Ccap - 1).long() # [N, num_experts]
            one_hot_pos = F.one_hot(pos_idx, num_classes = Ccap).to(probs.dtype) # [N, E, Ccap]

            combine = (expert_gate[:, None, None] *
                        expert_mask[:, :, None] *
                        one_hot_pos).to(X.dtype)
            dispatch = (combine > 0).to(torch.bool)
            
            return dispatch, combine, z_loss, aux_loss
        else:
            N, C, H, W = X.shape
            E = self.num_experts

            # Capacity factor
            if self.training:
                capcity_factor = getattr(self, 'capcity_factor', 1.25)
            else : 
                capcity_factor = 1.50
            
            Ccap = max(1, math.ceil(capcity_factor * N / self.num_experts))
            
            # Uniform choice batch level
            perm = torch.randperm(N, device = X.device)
            expert_index = torch.arange(N, device = X.device) % E
            expert_index = expert_index[perm]
            expert_mask = F.one_hot(expert_index, num_classes = E).to(torch.float32) # [N, E]

            # Queue per expert
            position_in_exp = torch.cumsum(expert_mask, dim = 0) - 1.0 # [N, E]
            within_capacity = (position_in_exp < Ccap).to(expert_mask.dtype) # [N, E]
            expert_mask = expert_mask.to(expert_mask.dtype) * within_capacity # [N, E]

            expert_gate = expert_mask.sum(dim = -1) # [N]

            pos_idx = position_in_exp.clamp(min = 0, max = Ccap - 1).long() # [N, E]
            one_hot_pos = F.one_hot(pos_idx, num_classes = Ccap).to(expert_mask.dtype) # [N, E, Ccap]

            combine = (expert_gate[:, None, None] *
                        expert_mask[:, :, None] *
                        one_hot_pos).to(X.dtype)
            dispatch = (combine > 0).to(torch.bool)

            probs = F.softmax(logits.float(), dim = 1) # [N, E]
            aux_loss = self.aux_loss(expert_mask, probs)
            
        return dispatch, combine, z_loss, aux_loss



    def z_loss(self, logits):
        """
        Z loss

        Args:
            logits (torch.Tensor) -> Logits
            batch_size (int) -> Batch size
        """
        z_loss = torch.logsumexp(logits, dim = -1) # [B, C]
        z_loss = torch.mean(z_loss)

        return z_loss

    def aux_loss(self, masked_probs, probs):
        """
        Aux loss

        Args:
            masked_probs (torch.Tensor) -> Masked probabilities
            probs (torch.Tensor) -> Probabilities
        """
        masked_probs = masked_probs.to(probs.dtype)

        density_1 = torch.mean(masked_probs, dim = 0)
        density_2 = torch.mean(probs, dim = 0)

        lb_loss = torch.sum(density_1 * density_2) * self.num_experts

        return lb_loss