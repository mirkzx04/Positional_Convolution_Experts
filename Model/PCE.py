import torch
import torch.nn as nn

from torch.nn import LazyLinear

from einops import rearrange

from Model.Components.ConvExpert import ConvExpert
from Datasets_Classes.PatchExtractor import PatchExtractor

class PCENetwork(nn.Module):
    def __init__(self, 
                    inpt_channel,
                    num_experts,
                    layer_number,
                    patch_size,
                    router,
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

        self.router = router
        self.num_classes = num_classes
        self.enable_router_metrics = enable_router_metrics

        self.patch_extractor = PatchExtractor(patch_size)

        # Defines layers and your experts + final convolution of the layer 
        # Defines convolution for pixel projection for the SSP embeddigs, used in router
        self.layers = nn.ModuleList()
        self.final_conv = nn.ModuleList()        
        self.convs_proj = nn.ModuleList()
        self.thresholds = nn.ParameterList()

        self.hard_threshold_router = hard_threshold_router

        inpt_channel = inpt_channel # Start channel (3 or 2) + 4 of positional information
        out_channel = 8

        for l in range(layer_number):
            # Defines all convolution parts of the layer, including experts
            self.convs_proj.append(
                nn.Conv2d(
                    in_channels = inpt_channel,
                    out_channels = 8,
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

            if l % 2 == 0:
                out_channel *= 2 

            inpt_channel = out_channel + 4   

        self.linear_layer = LazyLinear(self.num_classes)
    
    def get_exp_scores(self,
                       B,
                       P,
                       C,
                       pH,
                       pW,
                       X_patches,
                       layer_idx):
        """
        Get experts scores from router

        Args : 
            B (int)-> Batch size
            P // -> Patches number
            C // -> Channel of feature map
            pH //  -> Patch height
            pW // -> Patch width
            X_patches (torch.tensor) -> Feature map, tensor shape (B, P, C, nH, nW)
            layer_idx (int) -> index of current layer

        Returns:
            exp_scores -> tensor (B, P, num_experts)
            where num_experts is the number of experts in the layer
        """
        # Reshape from (B,C,H,W) -> (BxC, H, W) and project pixel
        X_patches_reshape = X_patches.reshape(B*P, C, pH, pW)
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
            X_patches, h_patches, w_patches = self.patch_extractor(X)
            B, P, C, pH, pW = X_patches.shape

            # Take experts scores
            exp_scores = self.get_exp_scores(
                B,
                P,
                C,
                pH,
                pW,
                X_patches,
                layer_idx
            )

            output = None
            X_patches_flat = X_patches.reshape(B*P, C, pH, pW)
            for exp_idx, expert in enumerate(layer_experts):
                # Get expert score
                exp_score = exp_scores[:, :, exp_idx]

                # Applies expert at patch, reshape dimension

                out = expert(X_patches_flat)
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