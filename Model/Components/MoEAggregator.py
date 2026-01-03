import math
from collections import deque
import torch
import torch.nn.functional as F

class MoEAggregator:
    """
    Aggrega metriche per router/dispatch MoE con:
      - accumuli per-layer robusti (fp64)
      - EMA per smussare il rumore
      - istogramma online per i quantili dei gate (evita cat/allocazioni)
      - metriche di bilanciamento robuste (CV, Gini, entropia normalizzata)
      - ratio max/min con smoothing di Laplace
      - opzionale all-reduce in DDP

    Args
    ----
    num_layers : int
        Numero di layer MoE.
    num_experts : list[int]
        Numero di esperti per layer, lunghezza = num_layers.
    ema_beta : float
        Fattore EMA (0.0=nessuno smoothing, 0.9 suggerito).
    gate_hist_bins : int
        Numero di bin per istogramma dei gate (per quantili p10/p50/p90).
    gate_hist_min, gate_hist_max : float
        Range clamp per i gate accumulati nell’istogramma.
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

        # --------- Global accumulators (micro) ---------
        self.tot_patches = 0
        self.tot_dropped_patches = 0
        self.tot_processed_patches = 0
        self.tot_capacity = 0

        # --------- Per-layer accumulators (macro) ---------
        # Conteggio token assegnati per expert (sommati su tutti i batch)
        self.usage_counts_layers = [torch.zeros(E, dtype=torch.float64) for E in num_experts]
        # Slot usati per expert (quante posizioni di capacity occupate almeno una volta)
        self.slot_used_layers = [torch.zeros(E, dtype=torch.float64) for E in num_experts]
        # Somma delle capacity (per normalizzare slot_used)
        self.ccap_sum_layers = [0 for _ in range(num_layers)]

        # Somma dei gate (combine) per expert e #conteggi (per media)
        self.gate_sum_layers = [torch.zeros(E, dtype=torch.float64) for E in num_experts]
        self.gate_counts_layers = [torch.zeros(E, dtype=torch.float64) for E in num_experts]

        # Router logits std (somma/contatori) + EMA per layer
        self.logits_std_sum_layers = [0.0 for _ in range(num_layers)]
        self.logits_std_count_layers = [0 for _ in range(num_layers)]
        self.logits_std_ema_layers = [0.0 for _ in range(num_layers)]

        # Entropia normalizzata dei logits del router (somma/contatori) + EMA per layer
        self.spec_entropy_sum_layers = [0.0 for _ in range(num_layers)]
        self.spec_entropy_count_layers = [0 for _ in range(num_layers)]
        self.spec_entropy_ema_layers = [0.0 for _ in range(num_layers)]

        # --------- Gate distribution (quantili) con istogramma online ---------
        self.gate_hist_bins = int(gate_hist_bins)
        self.gate_hist_min = float(gate_hist_min)
        self.gate_hist_max = float(gate_hist_max)
        self.gate_hist_edges = torch.linspace(self.gate_hist_min, self.gate_hist_max, self.gate_hist_bins + 1)
        self.gate_hist_counts = torch.zeros(self.gate_hist_bins, dtype=torch.float64)

        # --------- EMA config ---------
        self.ema_beta = float(ema_beta)

    # ----------------------- Utils -----------------------

    @staticmethod
    def _ema_update(prev: float, x: float, beta: float) -> float:
        return beta * prev + (1.0 - beta) * x

    @staticmethod
    def _mmm(values):
        """Mean, max, min su lista/tensor di scalari; ritorna (mean, max, min)."""
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
        """
        Indice di Gini su distribuzione u (positiva).
        Ritorna ∈ [0,1], 0 = bilanciamento perfetto.
        """
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
        """Somma il tensore su tutti i rank se DDP è inizializzato."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

    # ----------------------- Update per batch/layer -----------------------

    @torch.no_grad()
    def update_layer(self, layer_idx: int, dispatch: torch.Tensor, combine: torch.Tensor,
                     logits_std: float | None, logits: torch.Tensor | None):
        """
        Aggiorna accumulatori per un layer su un batch.

        Args
        ----
        layer_idx : int
            Indice del layer MoE (0..num_layers-1).
        dispatch : Tensor [N, E, Ccap] (bool/0-1)
            Assegnazioni token→(expert,slot).
        combine : Tensor [N, E, Ccap] (float)
            Pesi di combinazione (gate) per token→(expert,slot).
        logits_std : float | None
            Std dei logits del router (già scalati con temperatura) per questo layer/batch.
        logits : Tensor | None  [..., E]
            Logits del router (pre-softmax) per entropia specifica del layer.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if self.num_experts[layer_idx] == 0:
            return

        dispatch = dispatch.detach().cpu()
        combine = combine.detach().cpu()

        N, E, Ccap = dispatch.shape

        # ------------ Dropping / capacity ------------
        assign = dispatch.view(N, -1).sum(dim=1)            # [N], numero di slot assegnati per token
        dropped = int((assign == 0).sum().item())
        processed = N - dropped

        self.tot_patches += N
        self.tot_dropped_patches += dropped
        self.tot_processed_patches += processed
        self.tot_capacity += E * Ccap

        # ------------ Usage per expert ------------
        # Token assegnati (almeno un slot) per expert
        token_to_exp = dispatch.any(dim=2).to(torch.float64)     # [N, E]
        self.usage_counts_layers[layer_idx] += token_to_exp.sum(dim=0)  # [E]

        # Slot usati per expert (se almeno un token ha occupato lo slot)
        slot_used = dispatch.any(dim=0).to(torch.float64)        # [E, Ccap]
        self.slot_used_layers[layer_idx] += slot_used.sum(dim=1) # [E]
        self.ccap_sum_layers[layer_idx] += Ccap

        # ------------ Gate per expert ------------
        self.gate_sum_layers[layer_idx] += combine.sum(dim=(0, 2)).to(torch.float64)        # [E]
        self.gate_counts_layers[layer_idx] += token_to_exp.sum(dim=0)                       # [E]

        # ------------ Distribuzione gate (quantili) con istogramma ------------
        # Gate per token = somma su (E, Ccap) dei pesi assegnati
        gates = combine.view(N, -1).sum(dim=1).to(torch.float32)                            # [N]
        # Clamp e bucketize
        g = gates.clamp(self.gate_hist_min, self.gate_hist_max).cpu()
        idx = torch.bucketize(g, self.gate_hist_edges) - 1
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

        # ------------ Entropia normalizzata dei logits del router ------------
        if logits is not None:
            # softmax in fp32 per stabilità, poi entropia per token
            probs = torch.softmax(logits.detach().to(torch.float32), dim=-1)
            per_token_entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)    # [...]
            # normalizzazione per log(E)
            E_logits = probs.shape[-1]
            avg_norm_entropy = (per_token_entropy / math.log(E_logits)).mean().item()

            self.spec_entropy_sum_layers[layer_idx] += float(avg_norm_entropy)
            self.spec_entropy_count_layers[layer_idx] += 1
            self.spec_entropy_ema_layers[layer_idx] = self._ema_update(
                self.spec_entropy_ema_layers[layer_idx], float(avg_norm_entropy), self.ema_beta
            )

    # ----------------------- Finalizzazione (epoch) -----------------------

    @torch.no_grad()
    def finalize(self):
        """
        Calcola metriche aggregate (per-epoch) e ritorna un dict pronto da loggare.
        Esegue all-reduce in DDP (se inizializzato).
        """
        # ---- DDP all-reduce per i tensori per-layer ----
        for t in [*self.usage_counts_layers, *self.slot_used_layers,
                  *self.gate_sum_layers, *self.gate_counts_layers]:
            self._ddp_allreduce_(t)

        # all-reduce istogramma dei gate
        self._ddp_allreduce_(self.gate_hist_counts)

        # ---- Global micro-averages ----
        drop_rate = self.tot_dropped_patches / max(1, self.tot_patches)
        capacity_efficiency = self.tot_processed_patches / max(1, self.tot_capacity)

        moe_layer_ids = [i for i, E in enumerate(self.num_experts) if E > 0]

        # ---- Per-layer summaries ----
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

        # (logits std sums/cnts/spec sums/cnts sono scalari per layer: non serve all-reduce extra)

        eps_min = 1e-6  # per smoothing Laplace su ratio

        for layer_idx, in moe_layer_ids:
            counts = self.usage_counts_layers[layer_idx]
            gsum   = self.gate_sum_layers[layer_idx]
            gcnt   = self.gate_counts_layers[layer_idx]
            
            E = counts.numel()
            if E == 0 or counts.sum().item() == 0.0:
                # Layer senza attività (o non ancora visto)
                entropy_norm_layers.append(0.0)
                cov_usage_layers.append(0.0)
                gini_layers.append(0.0)
                imbalance_layers.append(0.0)
                mean_capacity_ratio_layers.append(0.0)
                avg_gate_mean_layers.append(0.0)
                dead_layers.append(E)
                active_layers.append(0)
            else:
                # Distribuzione d'uso normalizzata
                usage_frac = (counts / counts.sum().to(torch.float64)).clamp_min(1e-12)  # [E]
                usage_frac = usage_frac / usage_frac.sum()  # safety

                # Entropia normalizzata
                ent = -(usage_frac * usage_frac.log()).sum()
                ent_norm = float((ent / math.log(E)).item())
                entropy_norm_layers.append(ent_norm)

                # Coeff. di variazione su frazioni (robusto a scala)
                mean_u = float(usage_frac.mean().item())
                std_u = float(usage_frac.std(unbiased=False).item())
                cov_usage_layers.append(std_u / max(mean_u, 1e-12))

                # Gini
                gini_layers.append(self._gini(usage_frac))

                # Imbalance ratio (max/min) con smoothing di Laplace (robusto a min→0)
                mn = float(usage_frac.min().item())
                mx = float(usage_frac.max().item())
                ratio = (mx + eps_min) / (mn + eps_min)
                imbalance_layers.append(float(ratio))

                # Capacity ratio medio (slot usati / slot totali)
                slots_used_sum = self.slot_used_layers[layer_idx]  # [E]
                ccap_sum = self.ccap_sum_layers[layer_idx]
                if ccap_sum > 0:
                    ratio_per_exp = (slots_used_sum / ccap_sum).to(torch.float64)
                    mean_capacity_ratio_layers.append(float(ratio_per_exp.mean().item()))
                else:
                    mean_capacity_ratio_layers.append(0.0)

                # Media dei gate per expert (somma/comparsa)
                avg_gate_vec = gsum / gcnt.clamp_min(1.0)
                avg_gate_mean_layers.append(float(avg_gate_vec.mean().item()))

                # Dead/Active experts
                dead_layers.append(int((usage_frac < 1e-6).sum().item()))
                active_layers.append(int((usage_frac >= 1e-6).sum().item()))

            # Logits std (media aritmetica) + EMA
            if self.logits_std_count_layers[layer_idx] > 0:
                avg_std = self.logits_std_sum_layers[layer_idx] / self.logits_std_count_layers[layer_idx]
                logits_std_layers.append(float(avg_std))
            else:
                logits_std_layers.append(0.0)

            logits_std_ema_layers.append(float(self.logits_std_ema_layers[layer_idx]))

            # Spec entropy (media aritmetica) + EMA
            if self.spec_entropy_count_layers[layer_idx] > 0:
                avg_spec = self.spec_entropy_sum_layers[layer_idx] / self.spec_entropy_count_layers[layer_idx]
                spec_entropy_layers.append(float(avg_spec))
            else:
                spec_entropy_layers.append(0.0)

            spec_entropy_ema_layers.append(float(self.spec_entropy_ema_layers[layer_idx]))

        # ---- Gate quantiles via istogramma cumulato ----
        if self.gate_hist_counts.sum() > 0:
            cdf = torch.cumsum(self.gate_hist_counts, dim=0)
            total = cdf[-1].clamp_min(1)
            def _q(p):
                tgt = p * total
                idx = torch.searchsorted(cdf, tgt)
                idx = int(idx.clamp(0, self.gate_hist_bins).item())
                idx = min(idx, self.gate_hist_bins)  # edge case
                # mappa idx→edge; se idx==bins usa l'ultimo edge
                if idx == self.gate_hist_bins:
                    return float(self.gate_hist_edges[-1].item())
                return float(self.gate_hist_edges[idx].item())

            p10_gate = _q(0.10)
            p50_gate = _q(0.50)
            p90_gate = _q(0.90)
        else:
            p10_gate = p50_gate = p90_gate = 0.0

        # ---- Riduzioni su liste ----
        ent_mean, ent_min, ent_max = self._mmm(entropy_norm_layers)
        cov_mean, cov_min, cov_max = self._mmm(cov_usage_layers)
        gini_mean, gini_max, gini_min = self._mmm(gini_layers)  # (order max/min invertito per coerenza output)
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
            # Global micro
            "drop_rate": drop_rate,
            "capacity_efficiency": capacity_efficiency,

            # Per-layer summary (Entropy)
            "entropy_norm_mean": ent_mean,

            # Router logits entropy (per-token) aggregata
            "spec_entropy_mean": spec_entropy_mean,
            "spec_entropy_ema_mean": spec_entropy_ema_mean,

            # Per-layer summary (CV, Gini)
            "cov_usage_mean": cov_mean,
            "gini_mean": gini_mean,

            # Per-layer summary (Imbalance)
            "imbalance_mean": imbalance_mean,
            "imbalance_max": imbalance_max,
            "imbalance_min": imbalance_min,

            # Per-layer summary (Capacity ratio)
            "mean_capacity_ratio_mean": mean_capacity_ratio_mean,

            # Per-layer summary (Gate)
            "avg_gate_mean": avg_gate_mean_mean,

            # Gate distribution (quantiles)
            "gate_p10": p10_gate,
            "gate_p50": p50_gate,
            "gate_p90": p90_gate,

            # Dead/Active experts
            "dead_mean": dead_mean,
            "active_mean": active_mean,

            # Logits std
            "logits_std_mean": logits_std_mean,
            "logits_std_ema_mean": logits_std_ema_mean,
        }

    # ----------------------- Reset degli accumulatori -----------------------

    def reset(self):
        # Global
        self.tot_patches = 0
        self.tot_dropped_patches = 0
        self.tot_processed_patches = 0
        self.tot_capacity = 0

        # Per-layer
        self.usage_counts_layers = [torch.zeros_like(c, dtype=torch.float64) for c in self.usage_counts_layers]
        self.slot_used_layers = [torch.zeros_like(c) for c in self.slot_used_layers]
        self.ccap_sum_layers = [0 for _ in self.ccap_sum_layers]

        self.gate_sum_layers = [torch.zeros_like(c) for c in self.gate_sum_layers]
        self.gate_counts_layers = [torch.zeros_like(c) for c in self.gate_counts_layers]

        self.logits_std_sum_layers = [0.0 for _ in range(self.num_layers)]
        self.logits_std_count_layers = [0 for _ in range(self.num_layers)]
        self.logits_std_ema_layers = [0.0 for _ in range(self.num_layers)]

        self.spec_entropy_sum_layers = [0.0 for _ in range(self.num_layers)]
        self.spec_entropy_count_layers = [0 for _ in range(self.num_layers)]
        self.spec_entropy_ema_layers = [0.0 for _ in range(self.num_layers)]

        # Gate histogram
        self.gate_hist_counts.zero_()
