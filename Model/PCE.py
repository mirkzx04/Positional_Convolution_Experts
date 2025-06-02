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
        """

        self.router = router
        self.out_channel_exp = out_cha_experts

        self.patch_extractor = PatchExtractor(patch_size)

        self.layers = []
        for l in range(layer_number):
            layer = [ConvExpert(kernel_size=kernel_sz_experts, out_channel=out_cha_experts) for e in range(num_experts)]
            self.layers.append(layer)

    def forward(self, X):
        """
        Forward method of PCE Network

        Args :
            X -> input of network, tensor (B,C,H,W)
        """

        for layer in self.layers:
            # Divides feature map / input img in patches
            X_patches = self.patch_extractor(X)
            B, P, C, pH, pW = X_patches.shape
            print(f'X_patches shape : {X_patches.shape}')

            # Get experts score for all patches
            exp_scores = self.router(X_patches)
            exp_scores = exp_scores.reshape(B,P,-1)
            print(f'exp_scores shape : {exp_scores.shape}')

            output = None
            for exp in range(len(self.layers[1])):
                # Get expert score
                exp_score = exp_scores[:, :, exp]

                # Applies expert at patch, reshape dimension
                out = self.layers[1][exp](X_patches.reshape(B*P, C, pH, pW))
                _, C_out, H_out, W_out = out.shape
                out = out.reshape(B,P, C_out, H_out, W_out)

                # Concatenation of the experts feature map with weighted sum
                if output is None:
                    output = torch.zeros(B, P, C_out, H_out, W_out)
                output += out * exp_score.unsqueeze(2).unsqueeze(3).unsqueeze(4)
                print(f'output experts : {exp} -> {out.shape}')

            # Reassamble patch in in a single image [B, nP, C, H ,W] -> [B, C, H, W]
            output = rearrange(
                output, 
                'b (h w) c ph pw -> b c (h ph) (w pw)',
                h = 8,
                w = 8
            )
            print(f'output feature map shape : {output.shape}')





        

