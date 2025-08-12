import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class PatchExtractor(nn.Module):
    def __init__(self, patch_size, num_frequencies = 8):
        super().__init__()
        self.patch_size = patch_size
        self.num_frequencies = num_frequencies

    def fourier_features(self, coords):
        """
        Computes Fourier feature embeddings for the given coordinates.
        This method applies a set of sinusoidal transformations (sine and cosine) to the input coordinates
        using exponentially increasing frequencies. The resulting Fourier features can be used to encode
        positional information, which is useful in various machine learning models, such as neural networks
        for vision or signal processing tasks.
        
        Args:
            coords (torch.Tensor): Input tensor of coordinates with shape (..., D), where D is the dimensionality
                of the coordinates.
        Returns:
            torch.Tensor: Tensor containing the Fourier feature embeddings with shape (..., D * num_frequencies * 2),
                where `num_frequencies` is the number of frequency bands used for the encoding.
        """
    
        freqs = torch.arange(self.num_frequencies).to(device = coords.device)
        freqs = 2 ** freqs

        coords = coords.unsqueeze(-1)
        angles = 2 * torch.pi * coords * freqs
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim = -1)

        return fourier.reshape(*fourier.shape[:-2], -1)

    def forward(self, image):
        """
        Dvides an image in patch and return coordinate (x,y) for every patch

        Args :
            image : tensor (B,C, H, W)

        returns :
            patches : tensor (B, num_patches, C, patch_size x patch_size)
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
            
        coords = []
        B, _, H, W = image.shape

        # Applied padding
        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)

        H += pad_h
        W += pad_w

        # Extract patch
        patches = rearrange(image,
                    'b c (h p1) (w p2) -> b (h w) c p1 p2',
                    p1=self.patch_size, p2=self.patch_size)
        
        # Compute number of patch for dimension
        B = patches.shape[0]
        num_patches = patches.shape[1]
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # Extract coordinates (x,y) for all patch
        for i in range(h_patches):
            for j in range(w_patches):
                # Normalize coordinates to [0,1] range
                x_norm = j / (w_patches -1) if w_patches > 1 else 0.0
                y_norm = i / (h_patches -1) if h_patches > 1 else 0.0

                coords.append([x_norm, y_norm]) # (x,y) normalizede 

        coords = torch.tensor(coords, dtype=torch.float32, device=image.device)
        coords = repeat(coords, 'n xy -> b n coord xy', b = B)
        
        # Get coords fourier
        coords_fourier = self.fourier_features(coords)
        patch_pos_feats = torch.cat([coords, coords_fourier], dim = -1)
        patch_pos_feats = patch_pos_feats.unsqueeze(-1).unsqueeze(-1).exp(-1, -1, -1, self.patch_size, self.patch_size)


        # Add pixel-level position within patch
        yy,xx = torch.meshgrid(
            torch.linspace(0, 1, self.patch_size, device = image.device),
            torch.linspace(0, 1, self.patch_size, device = image.device),
            indexing='ij'
        )

        xx = xx.unsqueeze(0).unsqueeze(0).expand(B, num_patches, -1, -1)
        yy = yy.unsqueeze(0).unsqueeze(0).expand(B, num_patches, -1, -1)

        xx_feat = xx.unsqueeze(2)
        yy_feat = yy.unsqueeze(2)

        # Get xx and yy fourier
        xx_fourier = self.fourier_features(xx.unsqueeze(-1))
        yy_fourier = self.fourier_features(yy.unsqueeze(-1))

        xx_fourier = xx_fourier.permute(0, 1, 4, 2, 3)
        yy_fourier = yy_fourier.permute(0, 1, 4, 2, 3)

        pixel_feats = torch.can_cast([xx_feat, yy_feat, xx_fourier, yy_fourier], dim = 2)

        patches = patches.to(image.device)
        patches_with_coords = torch.cat([patches, patch_pos_feats, pixel_feats], dim = 2)

        return patches, patches_with_coords, h_patches, w_patches