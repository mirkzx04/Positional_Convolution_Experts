import math
from collections import deque  # (kept if you later need rolling windows)
import torch
import torch.nn.functional as F


class MoEAggregator:
    """
    Aggregates routing/dispatch metrics for MoE layers.

    Key points:
    - Device-aware: per-layer accumulators are moved to the input device on first update.
    - Uses fp64 for stable accumulation.
    - Online histogram for gate quantiles (p10/p50/p90) without storing all samples.
    - Multiple balance metrics: normalized entropy, CoV, Gini, max/min ratio.
    - Optional DDP all-reduce on finalize().
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: list[int],
        ema_beta: float = 0.9,
        gate_hist_bins: int = 64,
        gate_hist_min: float = 0.0,
        gate_hist_max: float = 1.0,
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts

        # --------- Global (per-epoch) counters ---------
        self.tot_patches = 0
        self.tot_dropped_patches = 0
        self.tot_processed_patches = 0
        self.tot_capacity = 0

        # --------- Per-layer accumulators (initialized on CPU; moved on first update) ---------
        self.usage_counts_layers = [torch.zeros(E, dtype=torch.float64) for E in num_experts]  # token usage per expert
        self.slot_used_layers = [torch.zeros(E, dtype=torch.float64) for E in num_experts]      # occupied slots per expert
        self.ccap_sum_layers = [0 for _ in range(num_layers)]                                   # sum of capacity across batches

        self.gate_sum_layers = [torch.zeros(E, dtype=torch.float64) for E in num_experts]       # sum of gate weights per expert
        self.gate_counts_layers = [torch.zeros(E, dtype=torch.float64) for E in num_experts]    # count of tokens per expert

        self.logits_std_sum_layers = [0.0 for _ in range(num_layers)]
        self.logits_std_count_layers = [0 for _ in range(num_layers)]
        self.logits_std_ema_layers = [0.0 for _ in range(num_layers)]

        self.spec_entropy_sum_layers = [0.0 for _ in range(num_layers)]  # normalized entropy of router probs
        self.spec_entropy_count_layers = [0 for _ in range(num_layers)]
        self.spec_entropy_ema_layers = [0.0 for _ in range(num_layers)]

        # --------- Online histogram for gate distribution (to extract quantiles) ---------
        self.gate_hist_bins = int(gate_hist_bins)
        self.gate_hist_min = float(gate_hist_min)
        self.gate_hist_max = float(gate_hist_max)
        self.gate_hist_edges = torch.linspace(
            self.gate_hist_min, self.gate_hist_max, self.gate_hist_bins + 1, dtype=torch.float64
        )
        self.gate_hist_counts = torch.zeros(self.gate_hist_bins, dtype=torch.float64)

        # --------- EMA config ---------
        self.ema_beta = float(ema_beta)

        # --------- Device management ---------
        self._device = None  # set on first update() based on dispatch.device

    # ----------------------- Helpers -----------------------

    @staticmethod
    def _ema_update(prev: float, x: float, beta: float) -> float:
        return beta * prev + (1.0 - beta) * x

    @staticmethod
    def _mmm(values):
        """Return (mean, max, min) as floats for a list/tensor; safe for empty inputs."""
        if isinstance(values, (list, tuple)):
            if not values:
                return 0.0, 0.0, 0.0
            t = torch.tensor(values, dtype=torch.float64)
        elif isinstance(values, torch.Tensor):
            if values.numel() == 0:
                return 0.0, 0.0, 0.0
            t = values.to(dtype=torch.float64)
        else:
            return 0.0, 0.0, 0.0
        return float(t.mean().item()), float(t.max().item()), float(t.min().item())

    @staticmethod
    def _gini(u: torch.Tensor) -> float:
        """Gini coefficient on a non-negative 1D tensor (in float64 domain)."""
        if u.numel() == 0:
            return 0.0
        u = u.clamp_min(1e-12)
        E = u.numel()
        u_sorted = torch.sort(u)[0]
        coef = (2 * torch.arange(1, E + 1, device=u.device, dtype=torch.float64) - E - 1)
        g = coef.double().dot(u_sorted.double()) / (E * u.sum().double())
        return float(g.item())

    @staticmethod
    def _ddp_allreduce_(tensor: torch.Tensor):
        """SUM all-reduce if torch.distributed is initialized."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

    def _ensure_device(self, device: torch.device):
        """
        Move all registered tensors to the specified device on first update
        or if device changes between updates.
        """
        if self._device is not None and self._device == device:
            return
        self.usage_counts_layers = [t.to(device=device) for t in self.usage_counts_layers]
        self.slot_used_layers = [t.to(device=device) for t in self.slot_used_layers]
        self.gate_sum_layers = [t.to(device=device) for t in self.gate_sum_layers]
        self.gate_counts_layers = [t.to(device=device) for t in self.gate_counts_layers]
        self.gate_hist_edges = self.gate_hist_edges.to(device=device)
        self.gate_hist_counts = self.gate_hist_counts.to(device=device)
        self._device = device

    # ----------------------- Per-batch/per-layer update -----------------------

    @torch.no_grad()
    def update_layer(
        self,
        layer_idx: int,
        dispatch: torch.Tensor,    # [N, E, Ccap], bool/0-1; device = GPU/CPU
        combine: torch.Tensor,     # [N, E, Ccap], float; same device as dispatch
        logits_std: float | None,  # optional scalar
        logits: torch.Tensor | None  # optional [..., E] logits used to compute normalized entropy
    ):
        """
        Update accumulators for a single MoE layer on the current batch.

        Args
        ----
        layer_idx : index of the MoE layer (0..num_layers-1)
        dispatch  : [N, E, Ccap] boolean/0-1 dispatch tensor
        combine   : [N, E, Ccap] combining weights (same device as dispatch)
        logits_std: optional float (std of router logits)
        logits    : optional tensor of router logits [..., E] to compute normalized entropy
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if self.num_experts[layer_idx] == 0:
            return

        # Align all accumulators to the device of inputs
        self._ensure_device(dispatch.device)

        dispatch = dispatch.detach()
        combine = combine.detach()

        N, E, Ccap = dispatch.shape

        # ------------ Dropping / capacity ------------
        assign = dispatch.view(N, -1).sum(dim=1)  # tokens with at least one assignment
        dropped = int((assign == 0).sum().item())
        processed = N - dropped

        self.tot_patches += N
        self.tot_dropped_patches += dropped
        self.tot_processed_patches += processed
        self.tot_capacity += E * Ccap

        # ------------ Expert usage (tokens per expert) ------------
        token_to_exp = dispatch.any(dim=2).to(torch.float64)                   # [N, E] on current device
        self.usage_counts_layers[layer_idx] += token_to_exp.sum(dim=0)         # [E]

        # ------------ Slots used per expert ------------
        slot_used = dispatch.any(dim=0).to(torch.float64)                      # [E, Ccap]
        self.slot_used_layers[layer_idx] += slot_used.sum(dim=1)               # [E]
        self.ccap_sum_layers[layer_idx] += Ccap

        # ------------ Gate sums/counts per expert ------------
        self.gate_sum_layers[layer_idx] += combine.sum(dim=(0, 2)).to(torch.float64)  # [E]
        self.gate_counts_layers[layer_idx] += token_to_exp.sum(dim=0)                 # [E]

        # ------------ Gate distribution (histogram for quantiles) ------------
        # Here we collapse all slots per token to a single scalar gate weight.
        gates = combine.view(N, -1).sum(dim=1).to(torch.float32)               # [N]
        g = gates.clamp(self.gate_hist_min, self.gate_hist_max)                # keep on current device
        idx = torch.bucketize(g.to(dtype=self.gate_hist_edges.dtype), self.gate_hist_edges) - 1
        idx = idx.clamp(0, self.gate_hist_bins - 1)
        binc = torch.bincount(idx, minlength=self.gate_hist_bins).to(torch.float64)
        self.gate_hist_counts += binc

        # ------------ Logits std + EMA ------------
        if logits_std is not None:
            v = float(logits_std)
            self.logits_std_sum_layers[layer_idx] += v
            self.logits_std_count_layers[layer_idx] += 1
            self.logits_std_ema_layers[layer_idx] = self._ema_update(
                self.logits_std_ema_layers[layer_idx], v, self.ema_beta
            )

        # ------------ Normalized entropy of router probabilities ------------
        if logits is not None:
            probs = torch.softmax(logits.detach().to(torch.float32), dim=-1)
            per_token_entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)
            E_logits = probs.shape[-1]
            avg_norm_entropy = (per_token_entropy / math.log(E_logits)).mean().item()

            self.spec_entropy_sum_layers[layer_idx] += float(avg_norm_entropy)
            self.spec_entropy_count_layers[layer_idx] += 1
            self.spec_entropy_ema_layers[layer_idx] = self._ema_update(
                self.spec_entropy_ema_layers[layer_idx], float(avg_norm_entropy), self.ema_beta
            )

    # ----------------------- End-of-epoch aggregation -----------------------

    @torch.no_grad()
    def finalize(self):
        """
        Compute aggregated metrics (per-epoch) and return a dict.
        - Performs DDP all-reduce on tensor accumulators if distributed is initialized.
        - Does not reset internal state (call reset() explicitly after logging).
        """
        # DDP all-reduce for layer tensors
        for t in [*self.usage_counts_layers, *self.slot_used_layers,
                  *self.gate_sum_layers, *self.gate_counts_layers]:
            self._ddp_allreduce_(t)

        # All-reduce for the gate histogram
        self._ddp_allreduce_(self.gate_hist_counts)

        # Global micro-averages
        drop_rate = self.tot_dropped_patches / max(1, self.tot_patches)
        capacity_efficiency = self.tot_processed_patches / max(1, self.tot_capacity)

        moe_layer_ids = [i for i, E in enumerate(self.num_experts) if E > 0]

        # Per-layer summaries (as lists; later aggregated with _mmm)
        entropy_norm_layers = []
        cov_usage_layers = []
        gini_layers = []
        imbalance_layers = []
        mean_capacity_ratio_layers = []
        avg_gate_mean_layers = []
        dead_layers = []
        active_layers = []
        logits_std_layers = []
        spec_entropy_layers = []
        logits_std_ema_layers = []
        spec_entropy_ema_layers = []

        eps_min = 1e-6

        for layer_idx in moe_layer_ids:
            counts = self.usage_counts_layers[layer_idx]     # [E]
            gsum   = self.gate_sum_layers[layer_idx]         # [E]
            gcnt   = self.gate_counts_layers[layer_idx]      # [E]

            E = counts.numel()
            if E == 0 or counts.sum().item() == 0.0:
                # No traffic on this layer
                entropy_norm_layers.append(0.0)
                cov_usage_layers.append(0.0)
                gini_layers.append(0.0)
                imbalance_layers.append(0.0)
                mean_capacity_ratio_layers.append(0.0)
                avg_gate_mean_layers.append(0.0)
                dead_layers.append(E)
                active_layers.append(0)
            else:
                # Usage distribution (normalized)
                usage_frac = (counts / counts.sum().to(torch.float64)).clamp_min(1e-12)  # [E]
                usage_frac = usage_frac / usage_frac.sum()

                # Normalized entropy (0..1)
                ent = -(usage_frac * usage_frac.log()).sum()
                ent_norm = float((ent / math.log(E)).item())
                entropy_norm_layers.append(ent_norm)

                # Coefficient of variation of usage_frac
                mean_u = float(usage_frac.mean().item())
                std_u = float(usage_frac.std(unbiased=False).item())
                cov_usage_layers.append(std_u / max(mean_u, 1e-12))

                # Gini
                gini_layers.append(self._gini(usage_frac))

                # Max/min ratio (smoothed)
                mn = float(usage_frac.min().item())
                mx = float(usage_frac.max().item())
                ratio = (mx + eps_min) / (mn + eps_min)
                imbalance_layers.append(float(ratio))

                # Capacity utilization (average occupied slots / capacity)
                slots_used_sum = self.slot_used_layers[layer_idx]  # [E]
                ccap_sum = self.ccap_sum_layers[layer_idx]
                if ccap_sum > 0:
                    ratio_per_exp = (slots_used_sum / ccap_sum).to(torch.float64)
                    mean_capacity_ratio_layers.append(float(ratio_per_exp.mean().item()))
                else:
                    mean_capacity_ratio_layers.append(0.0)

                # Average gate per expert
                avg_gate_vec = gsum / gcnt.clamp_min(1.0)
                avg_gate_mean_layers.append(float(avg_gate_vec.mean().item()))

                # Dead/active experts (thresholded)
                dead_layers.append(int((usage_frac < 1e-6).sum().item()))
                active_layers.append(int((usage_frac >= 1e-6).sum().item()))

            # Logits std (mean over batches)
            if self.logits_std_count_layers[layer_idx] > 0:
                avg_std = self.logits_std_sum_layers[layer_idx] / self.logits_std_count_layers[layer_idx]
                logits_std_layers.append(float(avg_std))
            else:
                logits_std_layers.append(0.0)

            logits_std_ema_layers.append(float(self.logits_std_ema_layers[layer_idx]))

            # Router normalized entropy (mean over batches)
            if self.spec_entropy_count_layers[layer_idx] > 0:
                avg_spec = self.spec_entropy_sum_layers[layer_idx] / self.spec_entropy_count_layers[layer_idx]
                spec_entropy_layers.append(float(avg_spec))
            else:
                spec_entropy_layers.append(0.0)

            spec_entropy_ema_layers.append(float(self.spec_entropy_ema_layers[layer_idx]))

        # Gate quantiles via cumulative histogram
        if self.gate_hist_counts.sum() > 0:
            cdf = torch.cumsum(self.gate_hist_counts, dim=0)
            total = cdf[-1].clamp_min(1)

            def _q(p):
                tgt = p * total
                idx = torch.searchsorted(cdf, tgt)
                idx = int(idx.clamp(0, self.gate_hist_bins).item())
                idx = min(idx, self.gate_hist_bins)
                if idx == self.gate_hist_bins:
                    return float(self.gate_hist_edges[-1].item())
                return float(self.gate_hist_edges[idx].item())

            p10_gate = _q(0.10)
            p50_gate = _q(0.50)
            p90_gate = _q(0.90)
        else:
            p10_gate = p50_gate = p90_gate = 0.0

        # Aggregate lists into summary scalars
        ent_mean, ent_min, ent_max = self._mmm(entropy_norm_layers)
        cov_mean, cov_min, cov_max = self._mmm(cov_usage_layers)
        gini_mean, gini_max, gini_min = self._mmm(gini_layers)
        imbalance_mean, imbalance_min, imbalance_max = self._mmm(imbalance_layers)
        mean_capacity_ratio_mean, _, _ = self._mmm(mean_capacity_ratio_layers)
        avg_gate_mean_mean, _, _ = self._mmm(avg_gate_mean_layers)
        dead_mean, _, _ = self._mmm(dead_layers)
        active_mean, _, _ = self._mmm(active_layers)
        logits_std_mean, _, _ = self._mmm(logits_std_layers)
        logits_std_ema_mean, _, _ = self._mmm(logits_std_ema_layers)
        spec_entropy_mean, _, _ = self._mmm(spec_entropy_layers)
        spec_entropy_ema_mean, _, _ = self._mmm(spec_entropy_ema_layers)

        return {
            "drop_rate": drop_rate,
            "capacity_efficiency": capacity_efficiency,

            "entropy_norm_mean": ent_mean,
            "spec_entropy_mean": spec_entropy_mean,
            "spec_entropy_ema_mean": spec_entropy_ema_mean,

            "cov_usage_mean": cov_mean,
            "gini_mean": gini_mean,

            "imbalance_mean": imbalance_mean,
            "imbalance_max": imbalance_max,
            "imbalance_min": imbalance_min,

            "mean_capacity_ratio_mean": mean_capacity_ratio_mean,
            "avg_gate_mean": avg_gate_mean_mean,

            "gate_p10": p10_gate,
            "gate_p50": p50_gate,
            "gate_p90": p90_gate,

            "dead_mean": dead_mean,
            "active_mean": active_mean,

            "logits_std_mean": logits_std_mean,
            "logits_std_ema_mean": logits_std_ema_mean,
        }

    def reset(self):
        """Reset all per-epoch accumulators. Keeps the current device setting."""
        # Global counters
        self.tot_patches = 0
        self.tot_dropped_patches = 0
        self.tot_processed_patches = 0
        self.tot_capacity = 0

        # Per-layer tensors (re-init on current device if known)
        self.usage_counts_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.usage_counts_layers
        ]
        self.slot_used_layers = [
            torch.zeros_like(c, device=self._device or c.device)
            for c in self.slot_used_layers
        ]
        self.ccap_sum_layers = [0 for _ in self.ccap_sum_layers]

        self.gate_sum_layers = [
            torch.zeros_like(c, device=self._device or c.device)
            for c in self.gate_sum_layers
        ]
        self.gate_counts_layers = [
            torch.zeros_like(c, device=self._device or c.device)
            for c in self.gate_counts_layers
        ]

        self.logits_std_sum_layers = [0.0 for _ in range(self.num_layers)]
        self.logits_std_count_layers = [0 for _ in range(self.num_layers)]
        self.logits_std_ema_layers = [0.0 for _ in range(self.num_layers)]

        self.spec_entropy_sum_layers = [0.0 for _ in range(self.num_layers)]
        self.spec_entropy_count_layers = [0 for _ in range(self.num_layers)]
        self.spec_entropy_ema_layers = [0.0 for _ in range(self.num_layers)]

        # Gate histogram
        if self.gate_hist_counts is not None:
            self.gate_hist_counts.zero_()