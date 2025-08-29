import math

import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

from torch import nn

# from .RouterGate import RouterGate

class Router(nn.Module):
    def __init__(self,
                num_experts,
                num_layers,
                noise_epsilon = 1e-2,
                router_temp = 1.5,
                load_factor = 0.02,
                capcity_factor_train = 1.25,
                capcity_factor_eval = 1.50):
        super().__init__()
        """
        Router class

        Args:
            num_experts (int) -> Number of experts
            num_layers (int) -> Number of layers
            noise_epsilon (float) -> Noise epsilon
            router_temp (float) -> Router temperature
            load_factor (float) -> Load factor
            capcity_factor_train (float) -> Capacity factor for training
            capcity_factor_eval (float) -> Capacity factor for evaluation
        """
        self.num_experts = num_experts
        self.num_layers = num_layers
        
        self.load_factor = load_factor
        self.noise_epsilon = noise_epsilon
        self.capcity_factor_train = capcity_factor_train
        self.capcity_factor_eval = capcity_factor_eval

        self.router_temp = router_temp

    def forward(self, X, router_gate, current_epoch = None):
        """
        Router forward method

        Args:
            X (torch.Tensor, shape [B*P, C, H, W]) -> Input
            router_gate (torch.Tensor, shape [B*P, C, num_experts]) -> Router gate
        """

        N, C, H, W = X.shape

        logits = router_gate(X) # [N, num_experts]
        # Compute z loss
        z_loss = self.z_loss(logits)

        # Add noise to logits in training mode
        if self.training:
            noise = logits.new_empty(logits.shape).uniform_(1 - self.noise_epsilon, 1 + self.noise_epsilon)
            logits = logits + noise

        # Apply softmax to get probabilities
        probs = F.softmax(logits.float() / self.router_temp, dim = 1) # [N, num_experts]
        E = self.num_experts
        N = probs.shape[0]

        # Top-1 : Index of best expert 
        expert_idx = probs.argmax(dim = 1) # [N]
        expert_prob = probs.gather(dim = 1, index = expert_idx.unsqueeze(1)).squeeze(1) # [N]
        expert_mask = F.one_hot(expert_idx, num_classes = E).to(probs.dtype) # [N, E]

        # Compute aux loss
        aux_loss = self.aux_loss(expert_mask, probs)

        # Expert capacity
        capcity_factor = self.capcity_factor_train if self.training else self.capcity_factor_eval
        
        # Calculate maximum capacity per expert: max number of tokens each expert can process
        # Uses capacity factor to balance computational load and performance
        Ccap = max(1, math.ceil(capcity_factor * N / E))

        # === CAPACITY ENFORCEMENT: Limit the number of tokens per expert ===
        
        # Sort tokens by assigned expert probability (from highest to lowest)
        order = torch.argsort(expert_prob, dim = 0, descending = True) # [N]
        
        # Reorder expert mask according to probability order
        mask_sorted = expert_mask[order] # [N, E]
        
        # Calculate position of each token in its respective expert's queue
        # torch.cumsum counts how many preceding tokens are assigned to the same expert
        pos_sorted = torch.cumsum(mask_sorted, dim = 0) - 1.0 # [N, E]
        
        # Determine which tokens are within expert capacity (pos < Ccap)
        within_capacity = (pos_sorted < Ccap).to(probs.dtype) # [N, E]
        
        # Apply capacity constraint: keep only tokens within capacity
        mask_sorted = mask_sorted * within_capacity # [N, E]

        # === UNSORT: Restore original token order ===
        # Create index to restore original order
        unsort = torch.empty_like(order)
        unsort[order] = torch.arange(N, device = order.device)
        
        # Restore original order of mask with capacity constraints applied
        mask = mask_sorted[unsort] # [N, E]

        # === COMPUTE ROUTING WEIGHTS ===
        
        # Calculate gate weights: expert probability * indicator if token is accepted
        gate = expert_prob * mask.sum(dim = -1) # [N]
        
        # Calculate position index for each token in expert's queue
        pos_idx = pos_sorted.clamp(min = 0, max = Ccap - 1).long()[unsort] # [N, E]
        
        # Create one-hot encoding of positions for dispatching
        one_hot_pos = F.one_hot(pos_idx, num_classes = Ccap).to(probs.dtype) # [N, E, Ccap]

        # === FINAL TENSORS FOR EXPERT COMPUTATION ===
        
        # Combine: tensor to aggregate expert outputs (gate_weight * mask * position)
        combine = (gate[:, None, None] * mask[:, :, None] * one_hot_pos).to(X.dtype)
        
        # Dispatch: boolean mask indicating which (token, expert, position) combinations are active
        dispatch = (combine > 0).to(torch.bool)

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

        f = masked_probs.mean(dim = 0)
        p = probs.mean(dim = 0)

        lb_loss = torch.sum(f * p) * E * self.load_factor

        return lb_loss