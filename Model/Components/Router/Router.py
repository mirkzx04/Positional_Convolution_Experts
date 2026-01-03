import math

import torch
from torch.distributions import OneHotCategorical
import torch.nn.functional as F

from sklearn.cluster import KMeans

from torch import logit, nn

# from .RouterGate import RouterGate

class Router(nn.Module):
    def __init__(
        self,
        num_experts,
        num_layers,
        noise_epsilon = 1e-2,
        router_temp = 1.5,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 1.50,
        noise_std = 0.02,
        ):
        super().__init__()
        """
        Router class with staged training support.

        Args:
            num_experts (int): Number of experts.
            num_layers (int): Number of layers.
            noise_epsilon (float): Epsilon for noise stability.
            router_temp (float): Temperature for the router logits.
            capacity_factor_train (float): Capacity factor during training.
            capacity_factor_eval (float): Capacity factor during evaluation.
            noise_std (float): Standard deviation for the noise added to logits.
            Ccap (float): Capacity coefficient (unused in current logic, but kept for compatibility).
        """
        self.num_experts = num_experts
        self.num_layers = num_layers
        
        self.noise_epsilon = noise_epsilon
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.noise_std = noise_std

        self.router_temp = router_temp


    def forward(self, X, router_gate, current_epoch = None):
        """
        Router forward pass with automatic phase management based on epochs.

        Args:
            X (torch.Tensor): Input tensor of shape [N, C, H, W] (or flattened [B*P, ...]).
            router_gate (nn.Module): The gate module to compute logits.
            current_epoch (int, optional): Current training epoch to determine routing phase.
        
        Returns:
            dispatch (torch.Tensor): Dispatch tensor for routing.
            combine (torch.Tensor): Combine tensor for aggregating results.
            z_loss (torch.Tensor): Z-loss value.
            aux_loss (torch.Tensor): Auxiliary load balancing loss.
            logits_std (float): Standard deviation of logits (for monitoring).
            logits (torch.Tensor): Raw logits (detached, on CPU).
        """

        N, C, H, W = X.shape
        E = self.num_experts

        # Always compute logits for monitoring and potential use
        logits = router_gate(X).to(dtype=torch.float32) # [N, num_experts]
        logits_temp = logits / self.router_temp
        logits_temp = logits_temp.clamp(min = -10.0, max = 10.0)

        logits_std = logits_temp.detach().std().item()
        
        # Route based on current phase (Uniform < 30 epochs, Specialized >= 30 epochs)
        if current_epoch == None:
            return self._specialized_routing(X, logits_temp, logits_std)
        if current_epoch < 50:
            return self._uniform_routing(X, logits_temp, logits_std)
        else:
            return self._specialized_routing(X, logits_temp, logits_std)

    def _specialized_routing(self, X, logits, logits_std):
        """
        Specialized top-1 routing (Phase 2/3).
        
        Standard Mixture-of-Experts (MoE) routing where tokens are assigned to the 
        expert with the highest probability (Top-1), subject to capacity constraints.
        """
        N, C, H, W = X.shape
        E = self.num_experts
        
        z_loss = self.z_loss(logits)
        if self.train and self.noise_std > 0:
            noise = torch.rand_like(logits.float()) * self.noise_std
            logits = logits + noise
        
        probs = F.softmax(logits.float(), dim = 1) # [N, num_experts]
        

        N, E = probs.shape

        # Select the expert with the highest probability for each token
        expert_idx  = probs.argmax(dim=1)                         # [N]
        expert_prob = probs[torch.arange(N, device=probs.device), expert_idx]  # [N]
        expert_mask = F.one_hot(expert_idx, num_classes=E).to(probs.dtype)     # [N, E]

        # Calculate capacity per expert (must be an integer)
        cap_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        Ccap = int(max(1, math.ceil(cap_factor * N / E)))

        # Sort tokens by probability in descending order
        order = torch.argsort(expert_prob, dim=0, descending=True)            # [N], long
        mask_sorted = expert_mask.index_select(0, order).to(torch.long)       # [N, E], 0/1 long
        pos_sorted = torch.cumsum(mask_sorted, dim=0) - 1                     # [N, E], long; -1 for unselected

        # Only selected tokens have a valid position (>= 0)
        within_cap = (pos_sorted >= 0) & (pos_sorted < Ccap)                  # [N, E], bool
        mask_sorted = (mask_sorted.bool() & within_cap).to(mask_sorted.dtype) # [N, E], 0/1

        # Restore original order (inverse permutation)
        unsort = torch.empty_like(order)                                       # [N]
        unsort[order] = torch.arange(N, device=order.device, dtype=order.dtype)

        # Apply capacity mask in the original order
        mask = mask_sorted.index_select(0, unsort).to(expert_mask.dtype)       # [N, E], 0/1

        # Calculate auxiliary loss using load (post-capacity) and importance (pre-capacity)
        importance = probs                                       # [N, E]
        load = mask.float()                                # [N, E]
        aux_loss = self.aux_loss(load, importance)

        # Normalize weights per expert (stop-gradient on denominator)
        raw   = (mask * expert_prob.unsqueeze(1))                              # [N, E], fp32
        denom = raw.sum(dim=1, keepdim=True).clamp_min(1e-6).detach()          # [N, 1]
        w     = raw / denom                                                    # [N, E] ; sum_t w[t,e] ≈ 1

        # Create one-hot positions in sorted space, then restore original order
        pos_idx_sorted = pos_sorted.clamp(min=0, max=Ccap-1).long()            # [N, E], long
        pos_idx = pos_idx_sorted.index_select(0, unsort)                       # [N, E], long
        one_hot_pos = F.one_hot(pos_idx, num_classes=Ccap).to(mask.dtype)      # [N, E, Ccap]

        # Final output tensors
        combine  = (w[:, :, None] * one_hot_pos).to(X.dtype)                   # [N, E, Ccap]
        dispatch = (mask[:, :, None].bool() & one_hot_pos.bool())   

        return dispatch, combine, z_loss, aux_loss, logits_std, logits.detach().cpu()

    def _uniform_routing(self, X, logits, logits_std):
        """
        Uniform routing (Phase 1).
        
        Distributes tokens evenly across all experts without using router decisions.
        This allows experts to learn diverse features before routing specialization.
        """
        N, C, H, W = X.shape
        E = self.num_experts

        probs = F.softmax(logits.float(), dim = 1) 
        
        # Use router's capacity calculation but ignore logits
        capcity_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        Ccap = max(1, math.ceil(capcity_factor * N / E))
        
        # Create uniform distribution across experts
        expert_assignment = torch.arange(N, device=X.device) % E  # Round-robin assignment
        expert_mask = F.one_hot(expert_assignment, num_classes=E).to(X.dtype)  # [N, E]
        
        # Calculate positions within each expert (simple sequential)
        positions = torch.zeros(N, E, device=X.device, dtype=torch.long)
        for e in range(E):
            expert_tokens = (expert_assignment == e)
            expert_count = int(expert_tokens.sum().item())
            if expert_count > 0:
                positions[expert_tokens, e] = torch.arange(expert_count, device=X.device)
        
        # Apply capacity constraints
        within_capacity = (positions < Ccap).to(X.dtype)  # [N, E]
        mask = expert_mask * within_capacity  # [N, E]
        
        # Create position one-hot encoding
        pos_clamped = positions.clamp(min=0, max=Ccap-1)  # [N, E] 
        one_hot_pos = F.one_hot(pos_clamped, num_classes=Ccap).to(X.dtype)  # [N, E, Ccap]
        
        # Uniform gate weights (equal confidence)
        gate = torch.ones(N, device=X.device, dtype=X.dtype) / E  # [N]
        
        # Create dispatch and combine tensors
        combine = (gate[:, None, None] * mask[:, :, None] * one_hot_pos)  # [N, E, Ccap]
        dispatch = (combine > 0).to(torch.bool)  # [N, E, Ccap]
        
        aux_loss = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        z_loss = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        # div_loss = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        
        return dispatch, combine, z_loss, aux_loss, logits_std, logits.detach().cpu()

    def z_loss(self, logits):
        """
        Computes Z-loss to encourage smaller logits.

        Args:
            logits (torch.Tensor): Router logits.
        """

        return torch.mean(torch.square(torch.logsumexp(logits, dim = -1)))

    def aux_loss(self, masked_probs, probs):
        """
        Computes Auxiliary Load Balancing Loss.
        
        Encourages balanced load across experts.

        Args:
            masked_probs (torch.Tensor): Probabilities after masking (load).
            probs (torch.Tensor): Original probabilities (importance).
        """
        f = masked_probs.mean(dim = 0)
        p = probs.mean(dim =0)

        lb_loss = (f * p).sum() * self.num_experts

        return lb_loss
    
    def diverity_loss(self, logits):
        """
        Encourage different experts to have different activation patterns.

        Args: 
            logits (torch.Tensor): Router logits [N, num_experts].
        """

        # Compute expert correlation
        E = logits.shape[-1]
        logits_norm = F.normalize(logits, dim = 0, p = 2)
        correlation = torch.mm(logits_norm.t(), logits_norm) # [E, E]

        # Penalize high correlation between experts
        identity = torch.eye(E, device = logits.device)
        
        