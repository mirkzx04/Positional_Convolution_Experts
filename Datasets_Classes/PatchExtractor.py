import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class PatchExtractor(nn.Module):
    def __init__(self, patch_size, num_frequencies = 4):
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
    
        freq_bands = 2.0 ** torch.linspace(0, self.num_frequencies - 1, self.num_frequencies, device=coords.device)
        coords_scaled = coords.unsqueeze(-1) * freq_bands * torch.pi
        return torch.cat([torch.sin(coords_scaled), torch.cos(coords_scaled)], dim=-1).flatten(-2)

    def get_positional(self, h_patches, w_patches, B, img_device):
        # Get coords
        coords = self.get_coords(h_patches, w_patches, B, img_device)

        # Get coords fourier
        coords_fourier = self.fourier_features(coords)
        patch_pos_feats = torch.cat([coords, coords_fourier], dim = -1)
        patch_pos_feats = patch_pos_feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.patch_size, self.patch_size)

    def get_coords(self, h_patches, w_patches, B, img_device):
        # Create axis
        h_coords = torch.linspace(0.0, 1.0, h_patches, device=img_device)
        w_coords = torch.linspace(0.0, 1.0, w_patches, device= img_device)

        y_grid, x_grid = torch.meshgrid(h_coords, w_coords, indexing='ij')

        coords = torch.stack([x_grid, y_grid], dim = -1) # [h, w, 2]
        coords = coords.flatten(start_dim=0, end_dim=1) #[num_patches, 2]
 
        coords = torch.tensor(coords, dtype=torch.float32, device=img_device)
        coords = repeat(coords, 'n xy -> b n xy', b = B)

        return coords

    def get_patches(self, image):
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
        patches = patches.to(image.device)

        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        
        # return h_patches, w_patches, patch_pos_feats, patches
        return h_patches, w_patches, patches