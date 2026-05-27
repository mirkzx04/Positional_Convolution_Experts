import math

import torch


class MoEAggregator:
    """
    Aggregates routing metrics for the current expert->top-k dispatch.

    With this router, per-expert balance metrics are mostly fixed by construction.
    The useful signals are token coverage, token overlap, and logit sharpness.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: list[int],
        ema_beta: float = 0.9,
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._device = None

        # Per-layer coverage / overlap counters.
        self.token_count_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.processed_count_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.multi_count_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.assignment_sum_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.rel_delta_sum_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]
        self.rel_delta_count_layers = [torch.zeros(1, dtype=torch.float64) for _ in range(num_layers)]

        # Per-layer sharpness counters.
        self.logits_std_sum_layers = [0.0 for _ in range(num_layers)]
        self.logits_std_count_layers = [0 for _ in range(num_layers)]
        self.logits_temp_std_sum_layers = [0.0 for _ in range(num_layers)]
        self.logits_temp_std_count_layers = [0 for _ in range(num_layers)]

        self.token_logit_entropy_sum_layers = [0.0 for _ in range(num_layers)]
        self.token_logit_entropy_count_layers = [0 for _ in range(num_layers)]

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

        self.token_count_layers = [t.to(device=device) for t in self.token_count_layers]
        self.processed_count_layers = [t.to(device=device) for t in self.processed_count_layers]
        self.multi_count_layers = [t.to(device=device) for t in self.multi_count_layers]
        self.assignment_sum_layers = [t.to(device=device) for t in self.assignment_sum_layers]
        self.rel_delta_sum_layers = [t.to(device=device) for t in self.rel_delta_sum_layers]
        self.rel_delta_count_layers = [t.to(device=device) for t in self.rel_delta_count_layers]
        self._device = device

    @torch.no_grad()
    def update_layer(
        self,
        layer_idx: int,
        token_idx: torch.Tensor,
        num_tokens: int,
        logits_std: float | None,
        logits_temp_std: float | None,
        logits: torch.Tensor | None,
        rel_delta_sum: float | None = None,
        rel_delta_count: int = 0,
    ):
        """
        Update accumulators for a single MoE layer on the current batch.

        Args:
            layer_idx: index of the MoE layer.
            token_idx: flattened token ids selected by the router, with repetitions
                when one token is assigned to multiple experts.
            num_tokens: total number of tokens before routing.
            logits_std: std of router raw logits, used to track sharpness.
            logits_temp_std: std of temperature-scaled logits, closer to dispatch sharpness.
            logits: router logits [N, E], used to compute the legacy
                `token_logit_entropy_mean` metric on the expert->token axis.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if self.num_experts[layer_idx] == 0:
            return

        if token_idx.is_cuda:
            device = token_idx.device
        elif logits is not None:
            device = logits.device
        else:
            device = torch.device("cpu")

        self._ensure_device(device)

        N = int(num_tokens)
        token_idx = token_idx.detach()

        if token_idx.numel() > 0:
            assign = torch.bincount(token_idx, minlength=N).to(torch.float64)
        else:
            assign = torch.zeros(N, dtype=torch.float64, device=device)

        processed = assign > 0
        multi = assign > 1

        self.token_count_layers[layer_idx].add_(float(N))
        self.processed_count_layers[layer_idx] += processed.sum().to(torch.float64)
        self.multi_count_layers[layer_idx] += multi.sum().to(torch.float64)
        self.assignment_sum_layers[layer_idx] += assign.sum()

        if logits_std is not None:
            self.logits_std_sum_layers[layer_idx] += float(logits_std)
            self.logits_std_count_layers[layer_idx] += 1

        if logits_temp_std is not None:
            self.logits_temp_std_sum_layers[layer_idx] += float(logits_temp_std)
            self.logits_temp_std_count_layers[layer_idx] += 1

        if logits is not None:
            probs = torch.softmax(logits.detach().to(torch.float32), dim=0)
            num_tokens = probs.shape[0]
            if num_tokens > 1:
                per_expert_entropy = -(probs * (probs + 1e-9).log()).sum(dim=0)
                avg_norm_entropy = (per_expert_entropy / math.log(num_tokens)).mean().item()
            else:
                avg_norm_entropy = 0.0

            self.token_logit_entropy_sum_layers[layer_idx] += float(avg_norm_entropy)
            self.token_logit_entropy_count_layers[layer_idx] += 1

        if rel_delta_sum is not None and rel_delta_count > 0:
            self.rel_delta_sum_layers[layer_idx].add_(float(rel_delta_sum))
            self.rel_delta_count_layers[layer_idx].add_(float(rel_delta_count))

    @torch.no_grad()
    def finalize(self):
        """
        Compute aggregated metrics (per-epoch) and return a compact dict.
        """
        for t in [
            *self.token_count_layers,
            *self.processed_count_layers,
            *self.multi_count_layers,
            *self.assignment_sum_layers,
            *self.rel_delta_sum_layers,
            *self.rel_delta_count_layers,
        ]:
            self._ddp_allreduce_(t)

        moe_layer_ids = [i for i, E in enumerate(self.num_experts) if E > 0]

        multi_rate_layers = []
        experts_per_processed_token_layers = []
        logits_std_layers = []
        logits_temp_std_layers = []
        token_logit_entropy_layers = []
        mean_rel_delta_layers = []

        total_tokens = 0.0
        total_processed = 0.0

        for layer_idx in moe_layer_ids:
            token_count = float(self.token_count_layers[layer_idx].item())
            processed_count = float(self.processed_count_layers[layer_idx].item())
            multi_count = float(self.multi_count_layers[layer_idx].item())
            assignment_sum = float(self.assignment_sum_layers[layer_idx].item())
            rel_delta_sum = float(self.rel_delta_sum_layers[layer_idx].item())
            rel_delta_count = float(self.rel_delta_count_layers[layer_idx].item())

            total_tokens += token_count
            total_processed += processed_count

            if token_count > 0.0:
                multi_rate_layers.append(multi_count / token_count)
            else:
                multi_rate_layers.append(0.0)

            if processed_count > 0.0:
                experts_per_processed_token_layers.append(assignment_sum / processed_count)
            else:
                experts_per_processed_token_layers.append(0.0)

            if self.logits_std_count_layers[layer_idx] > 0:
                avg_std = self.logits_std_sum_layers[layer_idx] / self.logits_std_count_layers[layer_idx]
                logits_std_layers.append(float(avg_std))
            else:
                logits_std_layers.append(0.0)

            if self.logits_temp_std_count_layers[layer_idx] > 0:
                avg_temp_std = (
                    self.logits_temp_std_sum_layers[layer_idx]
                    / self.logits_temp_std_count_layers[layer_idx]
                )
                logits_temp_std_layers.append(float(avg_temp_std))
            else:
                logits_temp_std_layers.append(0.0)

            if self.token_logit_entropy_count_layers[layer_idx] > 0:
                avg_ent = (
                    self.token_logit_entropy_sum_layers[layer_idx]
                    / self.token_logit_entropy_count_layers[layer_idx]
                )
                token_logit_entropy_layers.append(float(avg_ent))
            else:
                token_logit_entropy_layers.append(0.0)

            if rel_delta_count > 0.0:
                mean_rel_delta_layers.append(rel_delta_sum / rel_delta_count)
            else:
                mean_rel_delta_layers.append(0.0)

        drop_rate = 1.0 - (total_processed / max(1.0, total_tokens))
        multi_assigned_token_rate, _, _ = self._mmm(multi_rate_layers)
        experts_per_processed_token_mean, _, _ = self._mmm(experts_per_processed_token_layers)
        logits_std_mean, _, _ = self._mmm(logits_std_layers)
        logits_temp_std_mean, _, _ = self._mmm(logits_temp_std_layers)
        token_logit_entropy_mean, _, _ = self._mmm(token_logit_entropy_layers)

        metrics = {
            "drop_rate": drop_rate,
            "multi_assigned_token_rate": multi_assigned_token_rate,
            "experts_per_processed_token_mean": experts_per_processed_token_mean,
            "token_logit_entropy_mean": token_logit_entropy_mean,
            "logits_std_mean": logits_std_mean,
            "logits_temp_std_mean": logits_temp_std_mean,
        }
        for moe_layer_idx, mean_rel_delta in enumerate(mean_rel_delta_layers):
            metrics[f"moe_layer_{moe_layer_idx}/mean_rel_delta"] = float(mean_rel_delta)
        return metrics

    def reset(self):
        """Reset all per-epoch accumulators. Keeps the current device setting."""
        self.token_count_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.token_count_layers
        ]
        self.processed_count_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.processed_count_layers
        ]
        self.multi_count_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.multi_count_layers
        ]
        self.assignment_sum_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.assignment_sum_layers
        ]
        self.rel_delta_sum_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.rel_delta_sum_layers
        ]
        self.rel_delta_count_layers = [
            torch.zeros_like(c, dtype=torch.float64, device=self._device or c.device)
            for c in self.rel_delta_count_layers
        ]

        self.logits_std_sum_layers = [0.0 for _ in range(self.num_layers)]
        self.logits_std_count_layers = [0 for _ in range(self.num_layers)]
        self.logits_temp_std_sum_layers = [0.0 for _ in range(self.num_layers)]
        self.logits_temp_std_count_layers = [0 for _ in range(self.num_layers)]

        self.token_logit_entropy_sum_layers = [0.0 for _ in range(self.num_layers)]
        self.token_logit_entropy_count_layers = [0 for _ in range(self.num_layers)]
