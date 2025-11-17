from math import pi
import math
from numpy.ma import count
from sympy import Line2D
import torch
from torch._decomp.decompositions import isin_sorting

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

        self.logits_std_sum_layers = [0.0 for _ in range(num_layers)]
        self.logits_std_count_layers = [0 for _ in range(num_layers)]

        self.spec_entropy_sum_layers = [0.0 for _ in range(num_layers)]
        self.spec_entropy_count_layers = [0.0 for _ in range(num_layers)]

        self.reservoir_size = 20000
        self.gate_sample = []
    
    @torch.no_grad()
    def update_layer(self, layer_idx, dispatch, combine, logits_std, logits):
        dispatch, combine = dispatch.detach().cpu(), combine.detach().cpu()
        N, E, Ccap = dispatch.shape

        # Get assign patches
        assign = dispatch.view(N, -1).sum(dim = 1)
        dropped = int((assign == 0).sum().item()) # Number of dropped patches
        processed = N - dropped # Number of processed patches

        # print(f'Patches = {N}, dropped = {dropped}, processed = {processed}')
        # print(f'Drop rate : {dropped / N :.1%}')

        # Update global accumulators
        self.tot_patches += N
        self.tot_dropped_patches += dropped
        self.tot_processed_patches += processed
        self.tot_capacity += E * Ccap

        
        # Update per layer usage counts
        token_to_exp = dispatch.any(dim = 2).to(torch.float64) # [N, E]
        old_sum = self.usage_counts_layers[layer_idx].sum().item()
        self.usage_counts_layers[layer_idx] += token_to_exp.sum(dim = 0) # Percentage of patches assigned to each expert [E]

        # print(f'patches_to_exp.shape : {token_to_exp.shape}, sum : {token_to_exp.sum().item()}')
        # print(f'usage_increment : {token_to_exp.sum(dim = 0).tolist()}')

        slot_used = dispatch.any(dim=0).to(torch.float64) # [E, Ccap]
        self.slot_used_layers[layer_idx] += slot_used.sum(dim = 1) # Percentage of slots used per expert [E]
        self.ccap_sum_layers[layer_idx] += Ccap 
        
        # Update per layer gate sum & counts
        self.gate_sum_layers[layer_idx] += combine.sum(dim = (0, 2)).to(torch.float64) # Sum of gates per expert [E]
        # print(f' gate_increment : {combine.sum(dim = (0, 2)).to(torch.float64).tolist()}')
        self.gate_counts_layers[layer_idx] += token_to_exp.sum(dim = 0) # Count of gates per expert [E]

        # print(f' current _usage_counts[{layer_idx}] : {self.usage_counts_layers[layer_idx].tolist()}')
        # print(f' gat:sum[{layer_idx}] : {self.gate_sum_layers[layer_idx].tolist()}')
        # print(f' gate_counts[{layer_idx}] : {self.gate_counts_layers[layer_idx].tolist()}')

        gates = combine.view(N, -1).sum(dim = 1).to(torch.float64) # [N]
        if len(self.gate_sample) < self.reservoir_size:
            self.gate_sample.append(gates)

        if logits_std is not None:
            self.logits_std_sum_layers[layer_idx] += float(logits_std)
            self.logits_std_count_layers[layer_idx] += 1

            probs = torch.softmax(logits.to(torch.float32), dim = -1)

            per_token_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim = -1)
            avg_norm_entropy = (per_token_entropy / torch.log(torch.tensor(logits.shape[-1]))).mean()

            self.spec_entropy_sum_layers[layer_idx] += avg_norm_entropy.item()
            self.spec_entropy_count_layers[layer_idx] += 1

    
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
        dead_layers = []
        active_layers = []
        logits_std_layers = []
        spec_entropy_layers = []

        total_usage = sum(c.sum().item() for c in self.usage_counts_layers)
        print(f'total_usage = {total_usage:2f}')

        for layer_idx, (counts, gsum, gcnt) in enumerate(zip(self.usage_counts_layers, self.gate_sum_layers, self.gate_counts_layers)):
            E = counts.numel()
            counts_sum = counts.sum().item()
            gsum_sum = gsum.sum().item()
            gcnt_sum = gcnt.sum().item()


            # print(f' layer {layer_idx} : E={E}, counts_sum = {counts_sum}, gsum_sum = {gsum_sum}, gcnt_sum = {gcnt_sum}')
            if E == 0 or counts.sum() == 0:
                entropiy_norm_layers.append(0.0) 
                cov_usage_layers.append(0.0)
                imbalance_layers.append(0.0) 
                mean_capacity_ratio_layers.append(0.0) 
                avg_gate_mean_layers.append(0.0)

                dead_layers.append(E)
                active_layers.append(0)

                print(f'All are zero')

                continue

            dead_L = int((counts == 0).sum().item())
            active_L = int((counts > 0).sum().item())
            dead_layers.append(dead_L)
            active_layers.append(active_L)

            # Entropy
            usage_frac = counts / counts.sum().to(torch.float64) # Normalize
            p = usage_frac.clamp_min(1e-12)
            p = p / p.sum()
            ent = (- (p * p.log()).sum())
            ent_norm = ent / math.log(E)
            entropiy_norm_layers.append(float(ent_norm.item()))

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
                mean_capacity_ratio_layers.append(float(ration_per_exp.mean().item()))
            else:
                mean_capacity_ratio_layers.append(0.0)
            
            # avg gate per expert 
            avg_gate_vec = gsum / gcnt.clamp_min(1)
            avg_gate_mean_layers.append(float(avg_gate_vec.mean().item())) 

            if self.logits_std_sum_layers[layer_idx] > 0:
                avg_std = self.logits_std_sum_layers[layer_idx] / self.logits_std_count_layers[layer_idx]
                logits_std_layers.append(avg_std)
            else : 
                logits_std_layers.append(0.0)     

            if self.spec_entropy_count_layers[layer_idx] > 0:
                avg = self.spec_entropy_sum_layers[layer_idx] / self.spec_entropy_count_layers[layer_idx]
                spec_entropy_layers.append(avg)

        if len(self.gate_sample) > 0:
            gates_all = torch.cat(self.gate_sample, dim = 0).float().view(-1).cpu()
            p10_gate = float(gates_all.quantile(0.10).item())
            p50_gate = float(gates_all.quantile(0.50).item())
            p90_gate = float(gates_all.quantile(0.90).item())
        else:
            p10_gate = p50_gate = p90_gate = 0.0

        def _mmm(values):
            if isinstance(values, (list, tuple)):
                if not values:
                    return 0.0, 0.0, 0.0
                
                t = torch.tensor(values, dtype=torch.float64)
            elif isinstance(values, torch.Tensor):
                if values.numel() == 0:
                    return 0.0, 0.0, 0.0

                t = values.to(dtype=torch.float64)

            return float(t.mean().item()), float(t.max().item()), float(t.min().item())
        
        ent_mean, ent_min, ent_max = _mmm(entropiy_norm_layers)
        cov_mean, cov_min, cov_max = _mmm(cov_usage_layers)
        imbalance_mean, imbalance_min, imbalance_max = _mmm(imbalance_layers)
        mean_capacity_ratio_mean, mean_capacity_ratio_min, mean_capacity_ratio_max = _mmm(mean_capacity_ratio_layers)
        avg_gate_mean_mean, avg_gate_mean_min, avg_gate_mean_max = _mmm(avg_gate_mean_layers)
        dead_mean, dead_min, dead_max = _mmm(dead_layers)
        active_mean, active_min, active_max = _mmm(active_layers)
        logits_std_mean, logits_std_min, logits_std_max = _mmm(logits_std_layers)
        spec_entropy_mean, spec_entropy_min, spec_entropy_max = _mmm(spec_entropy_layers)

            
        return {
            # Global micro
            'drop_rate': drop_rate,
            'capacity_efficiency': capacity_efficiency,

            # Per layer summary (Entropy)
            'entropy_norm_mean' : ent_mean,

            'spec_entropy_mean' : spec_entropy_mean,

            # Per layer summary (Cov)
            'cov_usage_mean' : cov_mean,

            # Per layer summary (Imbalance)
            'imbalance_mean' : imbalance_mean,
            'imbalance_max' : imbalance_max,
            'imbalance_min' : imbalance_min,

            # Per layer summary (Capacity ratio)
            'mean_capacity_ratio_mean' : mean_capacity_ratio_mean,

            # Per layer summary (Gate)
            'avg_gate_mean' : avg_gate_mean_mean,

            # Per layer summary (Gate distribution)
            'gate_p10' : p10_gate,
            'gate_p50' : p50_gate,
            'gate_p90' : p90_gate,

            'dead_mean' : dead_mean,

            'active_mean' : active_mean,

            'logits_std_mean' : logits_std_mean,
        }

    def reset(self):
        # global
        self.tot_patches = 0
        self.tot_dropped_patches = 0
        self.tot_processed_patches = 0
        self.tot_capacity = 0

        E = self.num_experts

        # per-layer
        self.usage_counts_layers = [
            torch.zeros_like(c, dtype=torch.float64) for c in self.usage_counts_layers
        ]
        self.slot_used_layers = [
            torch.zeros_like(c) for c in self.slot_used_layers
        ]
        self.ccap_sum_layers = [0 for _ in self.ccap_sum_layers]
        self.gate_sum_layers = [
            torch.zeros_like(c) for c in self.gate_sum_layers
        ]
        self.gate_counts_layers = [
            torch.zeros_like(c) for c in self.gate_counts_layers
        ]

        self.spec_entropy_sum_layers = [0.0 for _ in range(self.num_layers)]
        self.spec_entropy_count_layers = [0.0 for _ in range(self.num_layers)]

        # reservoir
        self.gate_sample = []

        self.logits_std_sum_layers = [0.0 for _ in range(self.num_layers)]
        self.logits_std_count_layers = [0 for _ in range(self.num_layers)]
