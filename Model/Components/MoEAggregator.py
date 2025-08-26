import torch

class MoEAggregator():
    def __init__(self, num_layers, num_experts):
        """
        Initialize the MoEAggregator class.
        This class is used to aggregate the outputs of the experts.
        """
        self.num_layers = num_layers
        self.num_experts = num_experts

        # Global accumulators (micro)
        self.tot_patches = 0
        self.tot_dropped_patches = 0
        self.tot_processed_patches = 0
        self.tot_capacity = 0

        # per layer accumulators (macro)
        self.usage_counts_layers = [torch.zeros(E, dtype=torch.float64) for E in num_experts]
        self.slot_used_layers = [torch.zeros(E, dtype=torch.float64) for E in num_experts]

        # Per layer gate sum & counts
        self.ccap_sum_layers = [0 for _ in range(num_layers)]
        self.gate_sum_layers = [torch.zeros(E, dtype=torch.float64) for E in num_experts]
        self.gate_counts_layers = [torch.zeros(E, dtype=torch.float64) for E in num_experts]

        self.reservoir_size = 20000
        self.gate_sample = []
    
    @torch.no_grad()
    def update_layer(self, layer_idx, dispatch, combine):
        N, E, Ccap = dispatch.shape

        # Get assign patches
        assign = dispatch.view(N, -1).sum(dim = 1)
        dropped = int((assign == 0).sum().item()) # Number of dropped patches
        processed = N - dropped # Number of processed patches

        # Update global accumulators
        self.tot_patches += N
        self.tot_dropped_patches += dropped
        self.tot_processed_patches += processed
        self.tot_capacity += E * Ccap

        # Update per layer usage counts
        token_to_exp = dispatch.any(dim = 2).to(torch.float64) # [N, E]
        self.usage_counts_layers[layer_idx] += token_to_exp.sum(dim = 0) # Percentage of patches assigned to each expert [E]

        slot_used = dispatch.any(dim=0).to(torch.float64) # [E, Ccap]
        self.slot_used_layers[layer_idx] += slot_used.sum(dim = 1) # Percentage of slots used per expert [E]
        self.ccap_sum_layers[layer_idx] += Ccap 
        
        # Update per layer gate sum & counts
        self.gate_sum_layers[layer_idx] += combine.sum(dim = (0, 2)).to(torch.float64) # Sum of gates per expert [E]
        self.gate_counts_layers[layer_idx] += token_to_exp.sum(dim = 0) # Count of gates per expert [E]

        gates = combine.view(N, -1).sum(dim = 1).to(torch.float64) # [N]
        if len(self.gate_sample) < self.reservoir_size:
            self.gate_sample.append(gates)
    
    @torch.no_grad()
    def finalize(self):
        # Global micro-averages
        drop_rate = self.tot_dropped_patches / max(1, self.tot_patches)
        capacity_efficiency = self.tot_processed_patches / max(1, self.tot_capacity)

        # Entropy, cov, imbalace
        entropiy_norm_layers = []
        cov_usage_layers = []
        imbalance_layers = []
        mean_capacity_ratio_layers = []
        avg_gate_mean_layers = []

        for counts, gsum, gcnt in zip(self.usage_counts_layers, self.gate_sum_layers, self.gate_counts_layers):
            E = counts.numel()
            if E == 0 or counts.sum() == 0:
                entropies.append(0.0); covs.append(0.0); imbalances.append(0.0); mean_capacity_ratio_layers.append(0.0); avg_gate_mean_layers.append(0.0)

            # Entropy
            usage_frac = counts / counts.sum() # Normalize
            p = usage_frac + 1e-12
            p = p / p.sum()
            ent = (- (p * p.log()).sum()).item()
            ent_norm = ent / torch.log(torch.tensor(float(E)).item())
            entropiy_norm_layers.append(float(ent_norm))

            # Variational coefficient
            mean_u = counts.mean().item()
            std_u = counts.std().item()
            cov_usage = std_u / max(1e-12, mean_u)
            cov_usage_layers.append(float(cov_usage))

            # Imbalance index
            min_u = usage_frac.min().item()
            max_u = usage_frac.max().item()
            imbalance = (max_u / max(min_u, 1e-12)) if E > 1 else 0.0
            imbalance_layers.append(float(imbalance))

            # Capacity ratio per experts 
            slots_used_sum = self.slot_used_layers[layer_idx] # [E]
            ccap_sum = self.ccap_sum_layers[layer_idx] # Sum of capacity per step
            if ccap_sum > 0:
                ration_per_exp = (slots_used_sum / ccap_sum).to(torch.float64) # [E]
                mean_capacity_ration_layers.append(float(ration_per_exp.mean().item()))
            else:
                mean_capacity_ration_layers.append(0.0)
            
            # avg gate per expert 
            gsum = self.gate_sum_layers[layer_idx] 
            gcnt = self.gate_counts_layers[layer_idx].clamp_min(1)
            avg_gate_vec = gsum / gcnt
            avg_gate_mean_layers.append(float(avg_gate_vec.mean().item()))        

        if len(self.gate_sample) > 0:
            gates_all = torch.cat(self.gate_sample, dim = 0)
            p10_gate = float(gates_all.quantile(0.10).item())
            p50_gate = float(gates_all.quantile(0.50).item())
            p90_gate = float(gates_all.quantile(0.90).item())
        else:
            mean_gate = p10_gate = p50_gate = p90_gate = 0.0

        def _mmm(values):
            t = torch.tensor(values) if len(values) else torch.tensor([0.0])
            return float(t.mean().item()), float(t.max().item()), float(t.min().item())
        
        ent_mean, ent_min, ent_max = _mmm(entropiy_norm_layers)
        cov_mean, cov_min, cov_max = _mmm(cov_usage_layers)
        imbalance_mean, imbalance_min, imbalance_max = _mmm(imbalance_layers)
        mean_capacity_ratio_mean, mean_capacity_ratio_min, mean_capacity_ratio_max = _mmm(mean_capacity_ratio_layers)
        avg_gate_mean_mean, avg_gate_mean_min, avg_gate_mean_max = _mmm(avg_gate_mean_layers)


        return {
            # Global micro
            'drop_rate': drop_rate,
            'capacity_efficiency': capacity_efficiency,

            # Per layer summary (Entropy)
            'entropy_norm_mean' : ent_mean,
            'entropy_norm_max' : ent_max,
            'entropy_norm_min' : ent_min,

            # Per layer summary (Cov)
            'cov_usage_mean' : cov_mean,
            'cov_usage_max' : cov_max,
            'cov_usage_min' : cov_min,

            # Per layer summary (Imbalance)
            'imbalance_mean' : imbalance_mean,
            'imbalance_max' : imbalance_max,
            'imbalance_min' : imbalance_min,

            # Per layer summary (Capacity ratio)
            'mean_capacity_ratio_mean' : mean_capacity_ratio_mean,
            'mean_capacity_ratio_max' : mean_capacity_ratio_max,
            'mean_capacity_ratio_min' : mean_capacity_ratio_min,

            # Per layer summary (Gate)
            'avg_gate_mean' : avg_gate_mean_mean,
            'avg_gate_max' : avg_gate_mean_max,
            'avg_gate_min' : avg_gate_mean_min,

            # Per layer summary (Gate distribution)
            'gate_p10' : p10_gate,
            'gate_p50' : p50_gate,
            'gate_p90' : p90_gate,
        }

    def reset(self):
        # global
        self.tot_tokens = 0
        self.tot_dropped = 0
        self.tot_processed = 0
        self.tot_capacity = 0

        # per-layer
        self.usage_counts_layers = [
            torch.zeros_like(c) for c in self.usage_counts_layers
        ]
        self.slots_used_sum_layers = [
            torch.zeros_like(c) for c in self.slots_used_sum_layers
        ]
        self.ccap_sum_layers = [0 for _ in self.ccap_sum_layers]
        self.gate_sum_layers = [
            torch.zeros_like(c) for c in self.gate_sum_layers
        ]
        self.gate_cnt_layers = [
            torch.zeros_like(c) for c in self.gate_cnt_layers
        ]

        # reservoir
        self._gate_samples = []

