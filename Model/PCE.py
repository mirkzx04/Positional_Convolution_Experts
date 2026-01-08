import imp
import torch
import torch.nn as nn

from torch.nn import LazyLinear

from einops import rearrange
from timm.models.layers import DropPath

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

        self.drop_path = DropPath(0.1)
        
        last_channel = self.create_layers(
            num_experts=num_experts,
            dropout=dropout,
            layer_number=layer_number,
        )        

        self.router = Router(
            num_experts=num_experts,
            num_layers=layer_number,
            router_temp = router_temp,
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_val,
        )

        self.pooler = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(last_channel),
            nn.Dropout(0.1),
            nn.Linear(last_channel, num_classes),
            # nn.GELU(),
            # nn.Dropout(0.05),
            # nn.Linear(4 * last_channel, num_classes),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        num_experts_per_layer = [
            len(l.experts) if isinstance(l, PCELayer) else 0
            for l in self.layers
        ]
        self.moe_aggregator = MoEAggregator(
            num_layers=len(num_experts_per_layer),
            num_experts=num_experts_per_layer,
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
        inpt_channel = 32
        out_channel = 32

        for l in range(layer_number):
            current_gate_channel = inpt_channel + fourier_channel

            if l in [2, 4, 6]:
                transition_out = inpt_channel * 2
                self.layers.append(
                    DownsampleResBlock(in_ch = inpt_channel, out_ch= transition_out)
                )
                
                inpt_channel = transition_out
                out_channel = transition_out

                if l == 2 :
                    patch_size = patch_size // 4
                else : 
                    patch_size = max(1, patch_size // 2)
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

        img_device = X.device
        B = X.shape[0]

        X = self.stem(X)
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, DownsampleResBlock):
                X = layer(X)
            else :
                # Store layer attributes for cleaner access
                experts = layer.experts
                merge_gn = layer.merge_gn
                gamma = layer.gamma
                
                # Extract patches from the current feature map
                pre_layer = self.layers[layer_idx - 1] if layer_idx - 1 < len(self.layers) else None

                if isinstance(pre_layer, DownsampleResBlock) or pre_layer is None or layer_idx == 0: 
                    patch_size = layer.patch_size
                    self.patch_extractor.patch_size = patch_size

                    # Get token from X patches
                    h_patches, w_patches, X_patches = self.patch_extractor.get_patches(X)
                    P, C, H, W = X_patches.shape[1:] # Shape : [B, P, C, H, W]
                    N = B*P 

                    X_tokens = X_patches.reshape(B * P, C, H, W)

                    positional_features = self.patch_extractor.get_positional(h_patches, w_patches, B, img_device)
                    positional_features = positional_features.flatten(0, 1)

                    # tokens enriched with positional features 
                    X_positional = torch.cat([X_tokens, positional_features], dim = 1)
                else:
                    X_tokens = X # Shape : [N, C, H, W]
                
                # Get positional features from fourier
                
                
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

                # Extract routing indices for scatter operation
                n_idx, e_idx, c_idx = self._indices_from_dispatch(dispatch)
                
                # Distribute patches to their assigned expert slots
                if n_idx.numel() == 0:
                    X = X  # no-op, per chiarezza
                    continue
                
                # Process each expert independently on its assigned patches
                outputs = None
                per_exp_token_idx = [None] * E  # n_idx for each experts 
                per_exp_slot_idx  = [None] * E  # c_idx correspondents
                for e in range(E):
                    mask_e = (e_idx == e)
                    if mask_e.any():
                        per_exp_token_idx[e] = n_idx[mask_e]
                        per_exp_slot_idx[e]  = c_idx[mask_e]

                # Process only experts with tokens
                for e, expert in enumerate(experts):
                    n_e = per_exp_token_idx[e]
                    if n_e is None:
                        continue

                    # Tokens for experts e
                    x_e = X_tokens[n_e]  # [Ne, C, H, W]
                    y_e = expert(x_e)    # [Ne, C_out, H_out, W_out]
                    if outputs is None:
                        _, C_out, H_out, W_out = y_e.shape
                        outputs = X_tokens.new_zeros((N, C_out, H_out, W_out))
                    c_e = per_exp_slot_idx[e]                                  # [Ne]
                    w_e = combine[n_e, e, c_e].to(dtype=y_e.dtype).view(-1, 1, 1, 1)  # [Ne,1,1,1]

                    contrib = y_e * w_e  # [Ne, C_out, H_out, W_out]
                    outputs.index_add_(0, n_e, contrib)
                
                outputs = X_tokens + self.drop_path(gamma * outputs)
                
                # Reassemble patches back into spatial feature map
                next_layer = self.layers[layer_idx + 1] if layer_idx + 1 < len(self.layers) else None
                if next_layer is None or isinstance(next_layer, DownsampleResBlock):
                    X = rearrange(
                        outputs.view(B, -1, C_out, H_out, W_out), 
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