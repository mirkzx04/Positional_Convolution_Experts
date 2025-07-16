import torch
import torch.nn as nn

from torch.nn import LazyLinear

from einops import rearrange

from Model.Components.ConvExpert import ConvExpert
from Datasets_Classes.PatchExtractor import PatchExtractor

from Model.Components.Router import Router

class PCENetwork(nn.Module):
    def __init__(self, 
                    inpt_channel,
                    num_experts,
                    layer_number,
                    patch_size,
                    dropout,
                    num_classes,
                    hard_threshold_router = False,
                    enable_router_metrics = True,
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
        self.enable_router_metrics = enable_router_metrics

        self.patch_extractor = PatchExtractor(patch_size)
        self.router = Router(
            num_experts=num_experts,
            num_layers=layer_number,
        )

        # Defines layers and your experts + final convolution of the layer 
        # Defines convolution for pixel projection for the SSP embeddigs, used in router
        self.layers = nn.ModuleList()
        self.final_conv = nn.ModuleList()        
        self.convs_proj = nn.ModuleList()
        self.thresholds = nn.ParameterList()
        self.patches_sizes = []

        self.hard_threshold_router = hard_threshold_router

        inpt_channel = inpt_channel # Start channel (3 or 2) + 4 of positional information
        out_channel = 8

        self.create_layers(
            inpt_channel=inpt_channel,
            out_channel=out_channel,
            num_experts=num_experts,
            dropout=dropout,
            layer_number=layer_number,
        )        

        self.linear_layer = LazyLinear(self.num_classes)

    def create_layers(self, inpt_channel, out_channel, num_experts, dropout, layer_number):
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
        patch_size = self.patch_extractor.patch_size
        for l in range(layer_number):
            
            # Defines all convolution parts of the layer, including experts
            self.convs_proj.append(
                nn.Conv2d(
                    in_channels = inpt_channel,
                    out_channels = 16,
                    kernel_size=3,
                    padding=1,
                )
            )
            experts = nn.ModuleList([
                ConvExpert(
                    in_channel=inpt_channel,
                    out_channel=out_channel,
                    dropout=dropout
                )
                for _ in range(num_experts)
            ])
            self.layers.append(experts)

            self.final_conv.append(nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=1,
                padding=4
            ))

            # Defines threshold for experts scores
            self.thresholds.append(
                nn.Parameter(
                    torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
                )
            )
            self.patches_sizes.append(patch_size)

            patch_size = patch_size - 3 if patch_size - 3 >= 8 else patch_size

            inpt_channel = out_channel + 4   

            if l % 2 == 0:
                out_channel *= 2 

    def initialize_keys(self, X):
        for layer_idx in self.layers:
            # Get specific patch_size of layers and Patches
            X_patches, X_patches_reshape, h_patches, w_patches = self.get_patches(X)

            proj_patch = self.convs_proj[layer_idx](X_patches_reshape)
            self.router.initialize_keys(proj_patch)

            X = rearrange(
                X_patches,
                'b (h w) c ph pw -> b c (h ph) (w pw)',
                h = h_patches,
                w = w_patches
            )

    def get_patches(self, X, layer_idx):
        """
        Get patches and patches information

        Args:
            X (torch.Tensor) : Tensor of shape [B, C, H, W]
        """
        self.patch_extractor.patch_size = self.patches_sizes[layer_idx]
        X_patches, h_patches, w_patches = self.patch_extractor(X)
        B, P, C, pH, pW = X_patches.shape

        X_patches_reshape = X_patches.reshape(B*P, C, pH, pW)

        return X_patches, X_patches_reshape, h_patches, w_patches

    def get_exp_scores(self, X_patches_reshape, layer_idx, B, P):
        """
        Get experts scores from router

        Args : 
            X_patches (torch.tensor) -> Feature map, tensor shape (B * P, C, nH, nW)
            layer_idx (int) -> index of current layer

        Returns:
            exp_scores -> tensor (B, P, num_experts)
            where num_experts is the number of experts in the layer
        """

        X_patches_proj = self.convs_proj[layer_idx](X_patches_reshape)
        # Enable cache metric if rqeusted
        if self.enable_router_metrics:
            self.router.enable_metrics_cache()
            
        # get experts scores
        exp_scores = self.router(X_patches_proj, self.thresholds[layer_idx], self.hard_threshold_router)
        exp_scores = exp_scores.reshape(B, P, -1)

        return exp_scores

    def forward(self, X):
        """
        Forward method of PCE Network

        Args :
            X (torch.tensor) -> input of network, tensor.shape = (B,C,H,W)

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

        for layer_idx, layer_experts in enumerate(self.layers):
            # Divides feature map / input img in patches
            X_patches, X_patches_reshape, h_patches, w_patches = self.get_patches(X, layer_idx)
            B, P, _, _, _ = X_patches.shape

            # Take experts scores
            exp_scores = self.get_exp_scores(
                X_patches_reshape, layer_idx, B, P
            )

            output = None
            for exp_idx, expert in enumerate(layer_experts):
                # Get expert score
                exp_score = exp_scores[:, :, exp_idx]

                # Applies expert at patch, reshape dimension

                out = expert(X_patches_reshape)
                _, C_out, H_out, W_out = out.shape
                out = out.reshape(B,P, C_out, H_out, W_out)

                # Concatenation of the experts feature map with weighted sum
                if output is None:
                    output = torch.zeros(B, P, C_out, H_out, W_out, device=out.device, dtype=out.dtype)
                output += out * exp_score.unsqueeze(2).unsqueeze(3).unsqueeze(4)

            # Reassamble patch in in a single image [B, nP, C, H ,W] -> [B, C, H, W]
            # and applied final convolution 1x1 
            output = rearrange(
                output, 
                'b (h w) c ph pw -> b c (h ph) (w pw)',
                h = h_patches,
                w = w_patches
            )
            output = self.final_conv[layer_idx](output)

            X = output

        # Applying SSP at final experts output
        experts_output = X
        experts_spp_output = self.router.ssp(experts_output)
        logits = self.linear_layer(experts_spp_output)

        return logits