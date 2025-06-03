import torch
import torch.nn as nn

from einops import rearrange

from Model.Components.ConvExpert import ConvExpert
from Datasets_Classes.PatchExtractor import PatchExtractor

class PCENetwork(nn.Module):
    def __init__(self, 
                 num_experts,
                 kernel_sz_experts,
                 out_cha_experts,
                 layer_number,
                 patch_size,
                 router,
                 dropout
                 ):
        super().__init__()

        """
        Constructor of PCE Network

        Args : 
            kernel_sz_experts -> kernel size of experts
            out_cha_experts -> out channel for convolution in experts
            num_experts -> Number of experts per layer
            out_cha_router -> out channel for conv projection in router
            layer_number -> Numer of layers
            dropout -> dropout probability of the convolution experts
        """

        self.router = router
        self.out_channel_exp = out_cha_experts

        self.patch_extractor = PatchExtractor(patch_size)

        # Create layers and experts of layers
        self.layers = nn.ModuleList()
        self.final_conv = nn.ModuleList()
        self.convs_proj = nn.ModuleList()
        out_cha_key, in_channel = 7, 7
        for l in range(layer_number):
            experts_layer = nn.ModuleList([
                ConvExpert(
                    kernel_size=kernel_sz_experts,
                    in_channel=in_channel,
                    out_channel=out_cha_experts,
                    dropout=dropout
                    )
                for _ in range(num_experts)])
            self.layers.append(experts_layer)
            self.final_conv.append(
                nn.Conv2d(
                    kernel_size=1, 
                    in_channels=out_cha_experts, 
                    out_channels=out_cha_experts
                    )
                )
            self.convs_proj.append(
                nn.Conv2d(kernel_size=3, 
                    in_channels=7, 
                    out_channels=8
                    )
                )

            in_channel += 4
            out_cha_key += 4
            out_cha_experts += 4
            if l % 2 == 1:
                in_channel *= 2
                out_cha_experts *= 2
                out_cha_key *= 2

        print(f'expert in layers : {self.layers}')

        self.linear_layer = nn.Linear(out_cha_experts, out_features=10)
    
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
            B -> Batch size
            P -> Patches number
            C -> Channel of feature map
            pH -> Patch height
            pW -> Patch width
            X_patches -> Feature map, tensor (B, P, C, nH, nW)
            layer_idx -> index of current layer
        """
        # Reshape from (B,C,H,W) -> (BxC, H, W) and project pixel
        X_patches_reshape = X_patches.reshape(B*P, C, pH, pW)
        print(f'X_patches : {X_patches.shape}')
        X_patches_proj = self.convs_proj[layer_idx](X_patches_reshape)

        # get experts scores
        exp_scores = self.router(X_patches_proj)
        exp_scores = exp_scores.reshape(B,P,-1)

        return exp_scores


    def forward(self, X):
        """
        Forward method of PCE Network

        Args :
            X -> input of network, tensor (B,C,H,W)
        """

        for layer_idx, layer_experts in enumerate(self.layers):
            # Divides feature map / input img in patches
            X_patches = self.patch_extractor(X)
            B, P, C, pH, pW = X_patches.shape
            print(f'X_patches : {X_patches.shape}')

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
            for exp_idx, expert in enumerate(layer_experts):
                # Get expert score
                exp_score = exp_scores[:, :, exp_idx]

                print(f'Experts input : {X_patches.reshape(B*P, C, pH, pW).shape}')
                # Applies expert at patch, reshape dimension
                out = expert(X_patches.reshape(B*P, C, pH, pW))
                _, C_out, H_out, W_out = out.shape
                out = out.reshape(B,P, C_out, H_out, W_out)

                # Concatenation of the experts feature map with weighted sum
                if output is None:
                    output = torch.zeros(B, P, C_out, H_out, W_out)
                output += out * exp_score.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
            # Reassamble patch in in a single image [B, nP, C, H ,W] -> [B, C, H, W]
            # and applied final convolution 1x1
            output = rearrange(
                output, 
                'b (h w) c ph pw -> b c (h ph) (w pw)',
                h = 8,
                w = 8
            )
            print(f'Feature map reconstruction : {output.shape}')
            output = self.final_conv[layer_idx](output)
            print(f'output layer : {output.shape}')

            X = output

        logits = self.linear_layer(output)

        return logits





        

