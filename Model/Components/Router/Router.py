import math

import torch
from torch.distributions import OneHotCategorical
import torch.nn.functional as F

from torch import logit, nn
from dataclasses import dataclass

@dataclass(slots=True)
class RoutingState:
    token_idx: torch.Tensor
    expert_idx: torch.Tensor
    slot_idx: torch.Tensor
    weights: torch.Tensor
    num_tokens: int
    num_experts: int
    capacity: int

class Router(nn.Module):
    def __init__(
        self,
        num_experts,
        num_layers,
        router_temp = 1.5,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 1.50,
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
        
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.noise_std = 0

        self.router_temp = router_temp


    def forward(self, X, router_gate, positional_features, current_epoch = None):
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
            logits_std (float): Standard deviation of raw logits (for monitoring).
            logits_temp_std (float): Standard deviation of temperature-scaled logits.
            logits (torch.Tensor): Raw logits (detached, on CPU).
        """

        N = X.shape[0] # Total number of patches
        E = self.num_experts

        # Always compute logits for monitoring and potential use
        logits = router_gate(X, positional_features).to(dtype=torch.float32) # [N, Es]
        logits_std = logits.detach().std().item()

        logits_temp = logits / self.router_temp
        z_loss = self.z_loss(logits_temp)

        logits_temp_std = logits_temp.detach().std().item()
        logits_temp = logits_temp.clamp(min = -10.0, max = 10.0)

        
        # Route based on current phase (Uniform < 30 epochs, Specialized >= 30 epochs)
        if current_epoch == None:
            return self._specialized_routing(X, logits_temp, logits_std, logits_temp_std)
        if current_epoch < 10:
            return self._uniform_routing(X, logits_temp, logits_std, logits_temp_std)
        else:
            return self._specialized_routing(X, logits_temp, logits_std, logits_temp_std, z_loss)

    def _specialized_routing(self, X, logits, logits_std, logits_temp_std, z_loss):
        """
        Specialized top-1 routing (Phase 2/3).
        
        Standard Mixture-of-Experts (MoE) routing where tokens are assigned to the 
        expert with the highest probability (Top-1), subject to capacity constraints.
        """
        N = X.shape[0] # Total numbers of patches
        E = self.num_experts
        
        # Adding noise in logits 
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits.float()) * self.noise_std
            logits = logits + noise
        
        probs_e2t = F.softmax(logits.float(), dim = -1) # [N, E]
        div_loss = self.diverity_loss(probs_e2t)
        cov_loss = self.covarage_loss(probs_e2t)

        # Calculate capacity per expert (must be an integer)
        cap_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        ccap = math.ceil(cap_factor * N / E)
        k = min(max(1, int(ccap)), N)

        # Extract topk probs and topk index 
        topk_prob, topk_idx = torch.topk(
            probs_e2t, 
            k = k, 
            dim = 0,
            largest=True,
            sorted=True,
        )

        token_idx = topk_idx.reshape(-1)
        expert_idx = torch.arange(E, device=X.device).unsqueeze(0).expand(k, E).reshape(-1)
        slot_idx = torch.arange(k, device=X.device).unsqueeze(1).expand(k, E).reshape(-1)

        weights = topk_prob.reshape(-1).float()

        routing_state = RoutingState(
            token_idx=token_idx,
            expert_idx=expert_idx, 
            slot_idx=slot_idx,
            weights=weights,
            num_experts=E,
            num_tokens=N,
            capacity=k
        )

        return routing_state, cov_loss, z_loss, div_loss, logits_std, logits_temp_std, logits.detach()

    def _uniform_routing(self, X, logits, logits_std, logits_temp_std):
        """
        Uniform routing (Phase 1).
        
        Distributes tokens evenly across all experts without using router decisions.
        This allows experts to learn diverse features before routing specialization.
        """
        N = X.shape[0]
        E = self.num_experts

        cap_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        ccap = int(max(1, math.ceil(cap_factor * N / E)))

        token_idx = torch.arange(N, device=X.device)
        expert_idx = token_idx.remainder(E)
        slot_idx = torch.div(token_idx, E, rounding_mode="floor")

        keep = slot_idx < ccap

        routing_state = RoutingState(
            token_idx=token_idx[keep],
            expert_idx=expert_idx[keep],
            slot_idx=slot_idx[keep],
            weights=torch.full(
                (int(keep.sum().item()),),
                1.0,
                device=X.device,
                dtype=torch.float32,
            ),
            num_tokens=N,
            num_experts=E,
            capacity=ccap
        )

        z_loss = torch.tensor(0.0, device=X.device, dtype=torch.float32)
        div_loss = torch.tensor(0.0, device=X.device, dtype=torch.float32)
        cov_loss = torch.tensor(0.0, device=X.device, dtype=torch.float32)

        return routing_state, cov_loss, z_loss, div_loss, logits_std, logits_temp_std, logits.detach()
    
    def z_loss(self, logits):
        """
        Computes Z-loss to encourage smaller logits.

        Args:
            logits (torch.Tensor): Router logits.
        """

        return torch.logsumexp(logits, dim = -1).square().mean()

    # def aux_loss(self, masked_probs, probs):
    #     """
    #     Computes Auxiliary Load Balancing Loss.
        
    #     Encourages balanced load across experts.

    #     Args:
    #         masked_probs (torch.Tensor): Probabilities after masking (load).
    #         probs (torch.Tensor): Original probabilities (importance).
    #     """
    #     f = masked_probs.mean(dim = 0)
    #     p = probs.mean(dim =0)

    #     lb_loss = (f * p).sum() * self.num_experts

    #     return lb_loss
    
    def diverity_loss(self, probs):
        """
        Encourage different experts to have different different activation patterns.

        Args: 
            logits (torch.Tensor): Router logits [N, num_experts].
        """

        # Probs : [N, E]
        E = probs.shape[-1]
        if E <= 1:
            return probs.new_tensor(0.0)
        
        probs_norm = F.normalize(probs, dim = 0, p = 2) # Shape : [N, E]
        corr = probs_norm.t() @ probs_norm # Shape [E, E]
        I = torch.eye(E, device=probs.device, dtype=probs.dtype)
        return ((corr - I) ** 2).mean()

    def covarage_loss(self, probs_e2t):
        """
        Encourgates experts to not get the same tokens
        """
        return torch.tensor(0.0, device=probs_e2t.device, dtype=probs_e2t.dtype)
