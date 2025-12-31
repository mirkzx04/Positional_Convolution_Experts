import imp
import torch
import torch.nn as nn

from torch.nn import LazyLinear

from einops import rearrange

from Model.Components.Router import Router
from Model.Components.PCELayer import PCELayer
from Model.Components.DownsampleResBlock import DownsampleResBlock

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
                    router_temp,
                    noise_epsilon,
                    noise_std,
                    capacity_factor_train,
                    capacity_factor_val,
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

        self.pooler = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        last_channel = self.create_layers(
            num_experts=num_experts,
            dropout=dropout,
            layer_number=layer_number,
        )        

        self.router = Router(
            num_experts=num_experts,
            num_layers=layer_number,
            router_temp = router_temp,
            noise_epsilon=noise_epsilon,
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_val,
            noise_std=noise_std
        )
        print(last_channel)
        self.prediction_head = nn.Sequential(
            # nn.LayerNorm(last_channel),
            nn.Linear(last_channel, 4 * last_channel),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(4 * last_channel, num_classes),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True)
        )
        
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
            return 2 + 4 * F
        
        # Setting start parameters
        patch_size = self.patch_extractor.patch_size
        fourier_freq = self.patch_extractor.num_frequencies
        fourier_channel = get_fourie_channel(fourier_freq)
        inpt_channel = 8
        out_channel = 8

        for l in range(layer_number):
            current_gate_channel = inpt_channel + fourier_channel

            if l > 0 and l % 8 == 0:
                transition_out = inpt_channel * 2
                self.layers.append(
                    DownsampleResBlock(in_ch = inpt_channel, out_ch= out_channel)
                )
                
                inpt_channel = transition_out
                out_channel = transition_out
                patch_size = max(2, patch_size // 4) 

            else:
                self.layers.append(PCELayer(
                    inpt_channel=inpt_channel,
                    out_channel=out_channel,
                    num_experts=num_experts,
                    dropout=dropout,
                    patch_size=patch_size,
                    fourie_freq=fourier_freq,
                    gate_channel=current_gate_channel,
                    downsampling=False
                ))
                
        return out_channel

    def get_patches(self, X, patch_size):
        """
        Get patches and patches information

        Args:
            X (torch.Tensor) : Tensor of shape [B, C, H, W]
        """
        # Get patches
        self.patch_extractor.patch_size = patch_size
        h_patches, w_patches, X_positional, X_patches = self.patch_extractor(X)
        
        # Reshape X_patches
        B, P, C, pH, pW = X_patches.shape
        X_patches_reshape = X_patches.reshape(B*P, C, pH, pW)

        # Reshape X_patcches_coords
        B, P, C, pH, pW = X_positional.shape
        X_positional = X_positional.reshape(B*P, C, pH, pW)

        return h_patches, w_patches, X_positional, X_patches_reshape, X_patches

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
        tot_z_loss = 0.0
        tot_imb_loss = 0.0

        # Cache batch size to avoid repeated tensor accesses
        batch_size = X.shape[0]

        X = self.stem(X)
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.Sequential):
                X = layer(X)
            else :
                # Store layer attributes for cleaner access
                patch_size = layer.patch_size
                experts = layer.experts
                merge_gn = layer.merge_gn
                
                # Extract patches from the current feature map
                pre_layer = self.layers[layer_idx - 1] if layer_idx - 1 < len(self.layers) else None

                if isinstance(pre_layer, nn.Sequential) or pre_layer is None or layer_idx == 0: 
                    h_patches, w_patches, X_positional, X_patches_reshape, X_patches = self.get_patches(X, patch_size)
                    X_tokens = X_patches_reshape

                    # Unpack dimensions for easier reference
                    B, P, C, H, W = X_patches.shape 
                    N = B * P
                else :
                    X_tokens = X

                # Route patches to experts and compute auxiliary losses
                dispatch, combine, z_loss, imb_loss, logits_std, logits = self.router(
                    X_positional, 
                    layer.router_gate, 
                    current_epoch,
                )

                # Update aggregator statistics without gradients
                with torch.no_grad():
                    self.moe_aggregator.update_layer(
                        layer_idx, 
                        dispatch.detach(), 
                        combine.detach(), 
                        logits_std,
                        logits.detach()
                    )

                E, Ccap = dispatch.shape[1], dispatch.shape[2]
                
                # Allocate buffer for expert inputs with proper device and dtype
                expert_inputs = X_tokens.new_zeros((E, Ccap, C, H, W))

                # Extract routing indices for scatter operation
                n_idx, e_idx, c_idx = self._indices_from_dispatch(dispatch)
                
                # Distribute patches to their assigned expert slots
                if n_idx.numel() > 0:
                    expert_inputs[e_idx, c_idx] = X_tokens[n_idx]
                
                # Process each expert independently on its assigned patches
                expert_outputs_list = []
                
                for i, expert in enumerate(experts):
                    expert_outputs_list.append(expert(expert_inputs[i]))

                expert_outputs = torch.stack(expert_outputs_list, dim=0) 
                
                # Get output dimensions from processed patches
                _, _, C_out, H_out, W_out = expert_outputs.shape

                # Preallocate output buffer for gathering results
                outputs = X_tokens.new_zeros((N, C_out, H_out, W_out))
                
                if n_idx.numel() > 0:
                    # Apply routing weights to expert outputs
                    gates = combine[n_idx, e_idx, c_idx].to(dtype=expert_outputs.dtype).view(-1, 1, 1, 1)
                    
                    # Compute weighted contributions from each expert
                    contrib = expert_outputs[e_idx, c_idx] * gates
                    
                    # Accumulate contributions back to their original patch positions
                    outputs.index_add_(0, n_idx, contrib.to(outputs.dtype))

                # Reassemble patches back into spatial feature map
                next_layer = self.layers[layer_idx + 1] if layer_idx + 1 < len(self.layers) else None
                if next_layer is None or isinstance(next_layer, nn.Sequential):
                    X = rearrange(
                        outputs.view(batch_size, -1, C_out, H_out, W_out), 
                        'b (h w) c ph pw -> b c (h ph) (w pw)',
                        h=h_patches, w=w_patches
                    )
                else : 
                    X = outputs      

                X = merge_gn(X)
                tot_z_loss += z_loss
                tot_imb_loss += imb_loss

        # Apply global pooling and prediction head
        x = self.pooler(X) 
        x = self.flatten(x) 
        logits = self.prediction_head(x) 
        
        return logits, tot_z_loss, tot_imb_loss


    def _indices_from_dispatch(self, dispatch):
        """
        Get indices from dispatch

        Args:
            dispatch (torch.tensor) : tensor of shape [N, E, Ccap]
        """
        idx = dispatch.nonzero(as_tuple = False)
        device = dispatch.device
        return idx[:, 0], idx[:, 1], idx[:, 2]

    @torch.no_grad
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

        imbalance_index = (max_usage / max(min_usage, 1e-12)) if E > 1 else 0.0 # Compute imbalance index

        # Compute entropy and normalize it
        if E > 1:
            p = usage_frac + 1e-12
            p = p / p.sum() # Normalize 
            entropy = float((-(p * p.log()).sum().item())) # Compute shannon entropy
            entropy_norm = entropy / float(torch.log(torch.tensor(float(E))).item())

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