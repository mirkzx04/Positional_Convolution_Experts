import torch
import torch.nn as nn

from torch.nn import LazyLinear

from einops import rearrange

from Model.Components.ConvExpert import ConvExpert
from Model.Components.Router import Router
from Model.Components.PCELayer import PCELayer

from Model.Components.MoEAggregator import MoEAggregator
from Datasets_Classes.PatchExtractor import PatchExtractor
from Model.Components.Router.Router import Router

class PCENetwork(nn.Module):
    def __init__(self, 
                    num_experts,
                    layer_number,
                    patch_size,
                    dropout,
                    num_classes,
                 ):
        super().__init__()

        """
        Constructor of PCE Network

        Args : 
            kernel_sz_exps (int) -> kernel size of experts
            out_cha_exps // -> out channel for convolution in experts
            num_experts // -> Number of experts per layer
            out_cha_router // -> out channel for conv projection in router
            layer_number // -> Numer of layers
            dropout (float) -> dropout probability of the convolution experts
            patch_size (int) -> Size of patches, used in PatchExtractor
            router (Object.router) -> Router object, used for routing through experts
            threshold (float) -> Threshold for experts scores, used in router
        """

        self.num_classes = num_classes
        self.num_experts = num_experts

        self.patch_extractor = PatchExtractor(patch_size)

        self.layers = nn.ModuleList()

        self.pooler = nn.AdaptiveAvgPool2d(8)
        self.flatten = nn.Flatten()

        layer_info = self.create_layers(
            num_experts=num_experts,
            dropout=dropout,
            layer_number=layer_number,
        )        

        self.router = Router(
            num_experts=num_experts,
            num_layers=layer_number,
        )

        self.linear_layer = LazyLinear(self.num_classes)

        self.moe_aggregator = MoEAggregator(
            num_experts=[num_experts] * layer_number,
            num_layers=layer_number,
        )

    def create_layers(self, num_experts, dropout, layer_number):
        """
        Create layers of PCE Network

        Args:
            inpt_channel (int) -> Input channel of the first layer
            out_channel (int) -> Output channel of the first layer
            num_experts (int) -> Number of experts in each layer
            dropout (float) -> Dropout probability for experts
            layer_number (int) -> Number of layers in the network
        Returns:
            None
        """
        def get_fourie_channel(F):
            return 4 + 8 * F
        
        # Setting start parameters
        patch_size = self.patch_extractor.patch_size
        fourier_freq = self.patch_extractor.num_frequencies
        fourier_channel = get_fourie_channel(fourier_freq)
        inpt_channel = 3
        out_channel = 8

        layer_info = []
        for l in range(layer_number):
            # Gate channel is the sum of input channel and fourier channel
            gate_channel = inpt_channel + fourier_channel
            # Create PCE Layer
            self.layers.append(PCELayer(
                inpt_channel=inpt_channel,
                out_channel=out_channel,
                num_experts=num_experts,
                dropout=dropout,
                patch_size=patch_size,
                fourie_freq=fourier_freq,
                gate_channel=gate_channel,
            ))

            # Update patch size
            patch_size = patch_size - 2 if patch_size - 2 >= 8 else patch_size

            # Update input channel
            inpt_channel = out_channel

            # Update output channel
            if l % 2 == 0:
                out_channel *= 2 

        return layer_info
    
    def get_patches(self, X, patch_size):
        """
        Get patches and patches information

        Args:
            X (torch.Tensor) : Tensor of shape [B, C, H, W]
        """
        # Get patches
        self.patch_extractor.patch_size = patch_size
        X_patches, X_patches_coords, h_patches, w_patches = self.patch_extractor(X)

        # Reshape X_patches
        B, P, C, pH, pW = X_patches.shape
        X_patches_reshape = X_patches.reshape(B*P, C, pH, pW)

        # Reshape X_patcches_coords
        B, P, C, pH, pW = X_patches_coords.shape
        X_patches_coords_reshape = X_patches_coords.reshape(B*P, C, pH, pW)

        return X_patches, X_patches_reshape, X_patches_coords_reshape, h_patches, w_patches

    def forward(self, X, current_epoch=None):
        """
        Forward method of PCE Network

        Args :
            X (torch.tensor) -> input of network, tensor.shape = (B,C,H,W)
            current_epoch (int) -> current training epoch, used to determine routing strategy

        Pipeline of PCE Network:
            1. Divide input img/feature map in patches
            2. For each layer:
                2.1. Extract patches from input img/feature map
                2.2. Get experts scores from router
                2.3. For each expert:
                    2.3.1. Apply expert to patches
                    2.3.2. Concatenate experts outputs with weighted sum
                2.4. Reassemble patches in a single image
                2.5. Apply final convolution 1x1 to the output of the layer
            3. Create linear layer for classification if not exists
            4. Return logits and expert scores

        Returns:
            logits (torch.tensor) : tensor beatches (B, num_classes)
        """

        aux_loss = 0.0
        z_loss = 0.0

        tot_aux_loss = 0.0

        for layer_idx, layer in enumerate(self.layers):
            # Layer components
            experts = layer.experts
            patch_size = layer.patch_size
            router_gate = layer.router_gate

            expert_outputs_list = []

            # Divides feature map / input img in patches
            X_patches, X_patches_reshape, X_patches_coords_reshape, h_patches, w_patches \
            = self.get_patches(X, patch_size)
            B, P, C, H, W = X_patches.shape
            N = B * P

            dispatch, combine, z_loss, aux_loss = self.router(X_patches_coords_reshape, router_gate, current_epoch) # [N, E, Ccap], scalar, scalar
            self.moe_aggregator.update_layer(layer_idx, dispatch, combine)

            E, Ccap = dispatch.shape[1], dispatch.shape[2]

            # Patch expert_inputs : [E, Ccap, C, H, W]
            expert_inputs = X_patches_reshape.new_zeros((E, Ccap, C, H, W))

            n_idx, e_idx, c_idx = self._indices_from_dispatch(dispatch)
            if n_idx.numel() > 0:
                expert_inputs[e_idx, c_idx] = X_patches_reshape[n_idx]
            
            # Each expert is applied to its input
            for e , expert in enumerate(experts):
                # Expert inputs[e] : [Ccap, C, H, W] -> [Ccap, C, H, W]
                y_e = expert(expert_inputs[e])
                expert_outputs_list.append(y_e)
            expert_outputs = torch.stack(expert_outputs_list, dim = 0) # [E, Ccap, C, H, W]
            C_out, H_out, W_out = expert_outputs.shape[2], expert_outputs.shape[3], expert_outputs.shape[4]

            # Combine outputs : [E, C, H, W]
            outputs = X_patches_reshape.new_zeros((N, C_out, H_out, W_out))
            if n_idx.numel() > 0:
                gates = combine[n_idx, e_idx, c_idx].to(expert_outputs.dtype)
                contrib = expert_outputs[e_idx, c_idx] * gates.view(-1, 1, 1, 1)
                outputs.index_add_(0, n_idx, contrib)

            # Reassamble patch in in a single image [B, P, C, H ,W] -> [B, C, H, W]
            # and applied final convolution 1x1 
            outputs = outputs.reshape(B, P, C_out, H_out, W_out)
            output = rearrange(
                outputs, 
                'b (h w) c ph pw -> b c (h ph) (w pw)',
                h = h_patches,
                w = w_patches
            )

            X = output

            tot_aux_loss += 0.001 * z_loss + 0.05 * aux_loss

        # Applying SSP at final experts output
        experts_output = X
        experts_output_pooled = self.pooler(experts_output)
        experts_output_flatten = self.flatten(experts_output_pooled)
        logits = self.linear_layer(experts_output_flatten)

        return logits, aux_loss

    def _indices_from_dispatch(self, dispatch):
        """
        Get indices from dispatch

        Args:
            dispatch (torch.tensor) : tensor of shape [N, E, Ccap]
        """
        idx = dispatch.nonzero(as_tuple = False)
        device = dispatch.device
        return idx[:, 0], idx[:, 1], idx[:, 2]

    def _router_metrics(self, dispatch, combine, N, E, Ccap):
        """
        Get router metrics

        Args:
            dispatch (torch.tensor) : tensor of shape [N, E, Ccap]
            combine (torch.tensor) : tensor of shape [N, E, Ccap]
        """
        # Get assign patches
        assign_patches = dispatch.view(N, -1).sum(dim=1)
        dropped_mask = (assign_patches == 0)
        dropped_patches = dropped_mask.float().mean().item() # Percentage of dropped patches

        # Get processed patches
        processed_patches = int((assign_patches > 0).sum().item()) # Number of processed patches
        capacity_total = E * Ccap
        capacity_efficiency = (processed_patches / max(1, capacity_total)) # Percentage of capacity used

        # Get usage of experts
        patches_to_exp = dispatch.any(dim=2) # [N, E]
        usage_counts = patches_to_exp.sum(dim=0) # [E] Count of patches assigned to each expert
        usage_frac = (usage_counts.float() / max(1, N)) # [E] Percentage of patches assigned to each expert

        dead_experts = (int(usage_counts == 0).sum().item()) # Number of dead experts
        alive_experts = (int(usage_counts > 0).sum().item()) # Number of alive experts
        min_usage = float(usage_frac.min().item()) # Minimum usage of experts
        max_usage = float(usage_frac.max().item()) # Maximum usage of experts
        mean_usage = float(usage_frac.mean().item()) # Mean usage of experts

        imbalance_index = (max_uage / max(min_usage, 1e-12)) if E > 1 else 0.0 # Compute imbalance index

        # Compute entropy and normalize it
        if E > 1:
            p = usage_frac + 1e-12
            p = p / p.sum() # Normalize 
            entropy = float((-(p * p.log()).sum().item())) # Compute shannon entropy
            entropy_norm = entropy / float(torch.log(torch.tensor(float(E), device=device)).item())

        # Compute mean and std of usage of experts
        mean_u = float(usage_counts.float().mean().item()) if E > 1 else 0.0 # Mean number of patches assigned to each expert
        std_u = float(usage_counts.float().std().item()) if E > 1 else 0.0 # Standard deviation of number of patches assigned to each expert
        cov_usage = float(std_u / max(1e-12, mean_u)) if E > 1 else 0.0 # Covariance of number of patches assigned to each expert

        # Compute capacity usage of experts
        slot_used = dispatch.any(dim = 0) # [E, Ccap] 
        capacity_used = slot_used.sum(dim = 1).float() # number of slots used by each expert [E]
        capacity_ratio_per_exp = (capacity_used / max(1, Ccap)) # percentage of capacity used by each expert [E]
        mean_capacity_ratio = float(capacity_ratio_per_exp.mean().item()) # mean percentage of capacity used by each expert
        p95_capacity_ratio = float(capacity_ratio_per_exp.quantile(0.95).item()) # 95th percentile of percentage of capacity used by each expert

        # Collision (Saity check : 0.0)
        slot_counts = dispatch.sum(dim = 0) # [E, Ccap]
        collisions = int((slot_counts > 1).sum().item()) # number of collisions
        max_slot_count = int(slot_counts.max().item()) if slot_counts.numel() else 0 # maximum number of slots used by an expert
        if collisions >= 0:
            print(f'=== ALERT : COLLISIONS : {collisions} ===')

        # Gate per patches
        gate_per_patches = combine.view(N, -1).sum(dim = 1) # [N]
        mean_gate = float(gate_per_patches.mean().item()) # mean gate per patch
        p10_gate = float(gate_per_patches.quantile(0.10).item()) # 10th percentile of gate per patch
        p50_gate = float(gate_per_patches.quantile(0.50).item()) # 50th percentile of gate per patch
        p90_gate = float(gate_per_patches.quantile(0.90).item()) # 90th percentile of gate per patch

        # Gate per expert
        gate_per_exp_sum = combine.sum(dim = (0, 2)) # [E]
        counts = usage_counts.clamp_min(1)
        avg_gate_per_exp = float(gate_per_exp_sum / counts).float()
        mean_avg_gate_per_exp = float(avg_gate_per_exp.mean().item()) if E > 1 else 0.0
        min_avg_gate_per_exp = float(avg_gate_per_exp.min().item()) if E > 1 else 0.0 # Low : router has no confidence in expert
        max_avg_gate_per_exp = float(avg_gate_per_exp.max().item()) if E > 1 else 0.0 # High : router has confidence in expert

    
        
