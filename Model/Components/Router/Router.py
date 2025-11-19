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
        noise_std = 0.20
        ):
        super().__init__()
        """
        Router class with staged training support

        Args:
            num_experts (int) -> Number of experts
            num_layers (int) -> Number of layers
            noise_epsilon (float) -> Noise epsilon
            router_temp (float) -> Router temperature
            capacity_factor_train (float) -> Capacity factor for training
            capacity_factor_eval (float) -> Capacity factor for evaluation
            uniform_phase_ratio (float) -> Fraction of training for uniform routing (default: 0.2)
            blended_phase_ratio (float) -> Fraction of training for blended routing (default: 0.4)
            blending_schedule (str) -> Blending schedule type ('linear', 'cosine', 'exponential')
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
        Router forward with automatic phase management

        Args:
            X (torch.Tensor, shape [B*P, C, H, W]) -> Input
            router_gate (RouterGate) -> Router gate module
            current_epoch (int, optional) -> Current training epoch
            total_epochs (int, optional) -> Total training epochs
            force_specialized (bool) -> Force specialized routing (useful for validation)
        """

        N, C, H, W = X.shape
        E = self.num_experts

        # Always compute logits for monitoring and potential use
        logits = router_gate(X).to(dtype=torch.float32) # [N, num_experts]
        logits_std = logits.detach().std().item()

        # Clamp logits
        logits = logits.clamp(min = -10.0, max = 10.0)

        # Apply router temp
        logits = logits / self.router_temp
        
        # Calculate logits std for monitoring (detached from grad)
        
        # Route based on current phase
        if current_epoch <= 30:
            return self._uniform_routing(X, logits, logits_std)
        else:  # specialized
            return self._specialized_routing(X, logits, logits_std)

    def _specialized_routing(self, X, logits, logits_std):
        """
        Specialized top-1 routing (Phase 3)
        
        Standard MoE routing with expert specialization via top-1 assignment.
        """
        N, C, H, W = X.shape
        E = self.num_experts
        
        z_loss = self.z_loss(logits)
        diverity_loss = self.diverity_loss(logits)
        # z_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # Add noise to logits in training mode
        if self.training:
            noise = logits.new_empty(logits.shape).uniform_(1 - self.noise_epsilon, 1 + self.noise_epsilon)
            logits = logits * (noise * self.noise_std)

        # Apply softmax to get probabilities
        probs = F.softmax(logits.float(), dim = 1) # [N, num_experts]

        # Top-1 : Index of best expert 
        expert_idx = probs.argmax(dim = 1) # [N]
        expert_prob = probs.gather(dim = 1, index = expert_idx.unsqueeze(1)).squeeze(1) # [N]
        expert_mask = F.one_hot(expert_idx, num_classes = E).to(probs.dtype) # [N, E]

        # Compute aux loss
        aux_loss = self.aux_loss(expert_mask, probs)

        # Expert capacity
        capcity_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        
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
        within_capacity = (pos_sorted < Ccap).to(expert_mask.dtype) # [N, E]
        
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
        one_hot_pos = F.one_hot(pos_idx, num_classes = Ccap).to(expert_mask.dtype) # [N, E, Ccap]

        # === FINAL TENSORS FOR EXPERT COMPUTATION ===
        
        # Combine: tensor to aggregate expert outputs (gate_weight * mask * position)
        combine = (gate[:, None, None] * mask[:, :, None] * one_hot_pos).to(X.dtype)
        
        # Dispatch: boolean mask indicating which (token, expert, position) combinations are active
        dispatch = (combine > 0).to(torch.bool)

        return dispatch, combine, z_loss, aux_loss, diverity_loss, logits_std, logits.detach().cpu()

    def _uniform_routing(self, X, logits, logits_std):
        """
        Uniform routing (Phase 1)
        
        Distributes tokens evenly across all experts without using router decisions.
        This allows experts to learn diverse features before routing specialization.
        """
        N, C, H, W = X.shape
        E = self.num_experts
        
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
            expert_count = expert_tokens.sum().item()
            if expert_count > 0:
                expert_positions = torch.arange(expert_count, device=X.device)
                positions[expert_tokens, e] = expert_positions
        
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
        
        # Zero losses during uniform routing
        z_loss = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        aux_loss = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        
        return dispatch, combine, z_loss, aux_loss, logits_std, logits.detach().cpu()

    def z_loss(self, logits):
        """
        Z loss

        Args:
            logits (torch.Tensor) -> Logits
            batch_size (int) -> Batch size
        """

        return torch.mean(torch.square(torch.logsumexp(logits, dim = -1)))

    def aux_loss(self, masked_probs, probs):
        """
        Aux loss

        Args:
            masked_probs (torch.Tensor) -> Masked probabilities
            probs (torch.Tensor) -> Probabilities
        """
        f = masked_probs.mean(dim = 0)
        p = probs.mean(dim =0)

        lb_loss = torch.sum(f * p) * self.num_experts

        return lb_loss
    
    def diverity_loss(self, logits):
        """
        Encourgate different experts to have different activation patterns

        Args : 
            logits : Router logits [N, num_experts]
        """

        # Compute expert correlation
        E = logits.shape[-1]
        logits_norm = F.normalize(logits, dim = 0, p = 2)
        correlation = torch.mm(logits_norm.t(), logits_norm) # [E, E]

        # Penalize high correlation between experts
        identity = torch.eye(E, device = logits.device)
        off_diagonal = correlation * (1 - identity)
        diverity_loss = torch.sum(off_diagonal ** 2) / (E * (E - 1))

        return diverity_loss 
        
        