import torch
import torch.nn as nn

from torch.nn import LazyLinear

from einops import rearrange

from Model.Components.ConvExpert import ConvExpert
from Model.Components.Router import Router
from Model.Components.PCELayer import PCELayer

from Datasets_Classes.PatchExtractor import PatchExtractor


class PCENetwork(nn.Module):
    def __init__(self, 
                    inpt_channel,
                    num_experts,
                    layer_number,
                    patch_size,
                    dropout,
                    num_classes,
                    embed_dim,
                    hard_threshold_router = False,
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

        self.patch_extractor = PatchExtractor(patch_size)

        self.layers = nn.ModuleList()

        self.pooler = nn.AdaptiveAvgPool2d(8)
        self.flatten = nn.Flatten()

        self.hard_threshold_router = hard_threshold_router

        layer_info = self.create_layers(
            inpt_channel=inpt_channel, # Start channel (3 or 2) + 4 of positional information
            num_experts=num_experts,
            dropout=dropout,
            layer_number=layer_number,
            embed_dim = embed_dim
        )        

        self.router = Router(
            num_experts=num_experts,
            num_layers=layer_number,
            pce_layer_info = layer_info,
            embed_dim=embed_dim
        )

        self.linear_layer = LazyLinear(self.num_classes)

    def create_layers(self, num_experts, dropout, layer_number, embed_dim):
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
            embd_channel = inpt_channel + fourier_channel
            layer_info.append({
                'embd_channel': embd_channel, 
                'out_channel' : out_channel, 
                'patch_size': patch_size, 
                'embedd_dim': embed_dim})

            self.layers.append(PCELayer(
                inpt_channel=inpt_channel,
                out_channel=out_channel,
                num_experts=num_experts,
                dropout=dropout,
                patch_size=patch_size,
                fourie_freq=fourier_freq
            ))
            patch_size = patch_size - 3 if patch_size - 3 >= 8 else patch_size

            inpt_channel = out_channel

            if l % 2 == 0:
                out_channel *= 2 

        return layer_info
    
    def initialize_keys(self, X):
        """
        Initialize router keys with projection convolution

        Args:
            X (torch.Tensor) : Training set 
        """
        for layer_idx, layer in enumerate(self.layers):
            # # Get patches
            # X_patches, X_patches_reshape, h_patches, w_patches = self.get_patches(X, layer.patch_size)

            # # Check if X_patches_reshape has same channels as conv_proj
            # expected_in_channels = self.router.embedders[layer_idx].cnn[0].in_channels
            # current_channels = X_patches_reshape.shape[1]
            # if current_channels < expected_in_channels:
            #     diff = expected_in_channels - current_channels
            #     pad = torch.zeros(X_patches_reshape.size(0), diff, X_patches_reshape.size(2), X_patches_reshape.size(3),
            #                       device=X_patches_reshape.device, dtype=X_patches_reshape.dtype)
            #     X_patches_reshape = torch.cat([X_patches_reshape, pad], dim = 1)
            # elif current_channels > expected_in_channels:
            #     X_patches_reshape = X_patches_reshape[:, :expected_in_channels, :, :]

            self.router.initialize_keys(None, layer_idx)

            # Recompose the patches into the original image
            # X = rearrange(
            #     X_patches,
            #     'b (h w) c ph pw -> b c (h ph) (w pw)',
            #     h = h_patches,
            #     w = w_patches
            # )

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

    def get_exp_scores(self, X_patches_coords_reshape, layer_idx, B, P, thresholds):
        """
        Get experts scores from router

        Args : 
            X_patches (torch.tensor) -> Feature map, tensor shape (B * P, C, nH, nW)
            layer_idx (int) -> index of current layer

        Returns:
            exp_scores -> tensor (B, P, num_experts)
            where num_experts is the number of experts in the layer
        """ 
        # get experts scores
        exp_scores = self.router(X_patches_coords_reshape, layer_idx, thresholds, self.hard_threshold_router)
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

        for layer_idx, layer in enumerate(self.layers):
            # Layer components
            experts = layer.experts
            final_conv = layer.final_conv
            threshold = layer.threhsold
            patch_size = layer.patch_size

            num_expert = len(experts)

            # Divides feature map / input img in patches
            X_patches, X_patches_reshape, X_patches_coords_reshape, h_patches, w_patches \
            = self.get_patches(X, patch_size)
            B, P, _, _, _ = X_patches.shape

            # Take experts scores
            exp_scores = self.get_exp_scores(
                X_patches_coords_reshape, layer_idx, B, P, threshold
            )

            # Applied all experts at batch
            all_outputs = [expert(X_patches_reshape) for expert in experts]
            all_outputs = torch.stack(all_outputs, dim = 0) # Shape : [num_experts, B*P, C_out, H_out, W_out]

            # Reshape [num_experts, B*P, C_out, H_out, W_out] -> [B, P, num_experts, C_out, H_out, W_out]
            all_outputs = all_outputs.permute(1, 0, 2, 3, 4) # Shape : [B*P, num_experts, C_out, H_out, W_out]
            C_out, H_out, W_out = all_outputs.shape[2], all_outputs.shape[3], all_outputs.shape[4]
            all_outputs = all_outputs.reshape(B, P, num_expert, C_out, H_out, W_out)

            # Applied router scores
            exp_score = exp_scores.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) #Shape : [B, P, num_experts, 1,1,1]
            output = (all_outputs * exp_score).sum(dim = 2) # Shape : [B, P, C_out, H_out, W_out]

            # Reassamble patch in in a single image [B, P, C, H ,W] -> [B, C, H, W]
            # and applied final convolution 1x1 
            output = rearrange(
                output, 
                'b (h w) c ph pw -> b c (h ph) (w pw)',
                h = h_patches,
                w = w_patches
            )
            output = final_conv(output)

            X = output

        # Applying SSP at final experts output
        experts_output = X
        experts_output_pooled = self.pooler(experts_output)
        experts_output_flatten = self.flatten(experts_output_pooled)
        logits = self.linear_layer(experts_output_flatten)

        return logits