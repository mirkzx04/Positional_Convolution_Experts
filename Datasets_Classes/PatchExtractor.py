import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class PatchExtractor(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, image):
        """
        Dvides an image in patch and return coordinate (x,y) for every patch

        Args :
            image : tensor (B,C, H, W)

        returns :
            patches : tensor (B, num_patches, C, patch_size x patch_size)
        """
        coords = []
        B, C, H, W = image.shape

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
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # Extract coordinates (x,y) for all patch
        for i in range(h_patches):
            for j in range(w_patches):
                # Normalize coordinates to [0,1] range
                x_norm = j / (w_patches -1) if w_patches > 1 else 0.0
                y_norm = i / (h_patches -1) if h_patches > 1 else 0.0

                coords.append([x_norm, y_norm]) # (x,y) normalizede 

        coords = torch.tensor(coords, dtype=torch.float32)
        coords = repeat(coords, 'n coord -> b n coord h w', b = B, h = self.patch_size, w = self.patch_size)

        # Add pixel-level position within patch
        yy,xx = torch.meshgrid(
            torch.linspace(0, 1, self.patch_size),
            torch.linspace(0, 1, self.patch_size),
            indexing='ij'
        )

        # Shape: (1, 1, 1, p, p) → broadcast to (B, P, 1, p, p)
        xx = xx.unsqueeze(0).unsqueeze(0).unsqueeze(2).expand(B, patches.shape[1], 1, self.patch_size, self.patch_size)
        yy = yy.unsqueeze(0).unsqueeze(0).unsqueeze(2).expand(B, patches.shape[1], 1, self.patch_size, self.patch_size)

        # Concatenate all channels: original + patch coords + pixel coords
        patches = patches.to(image.device)
        patches_with_coords = torch.cat([patches, coords, xx, yy], dim=2)  # (B, P, C + 4, p, p)

        return patches_with_coords, h_patches, w_patches