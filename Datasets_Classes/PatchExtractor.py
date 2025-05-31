from einops import rearrange, repeat

import torch

class PatchExtractor:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def extract_patches_coords(self, image):
        """
        Dvides an image in patch and return coordinate (x,y) for every patch

        Args :
            image : tensor (B,C, H, W)

        returns :
            patches : tensor (B, num_patches, C, patch_size x patch_size)
        """
        coords = []
        B, C, H, W = image.shape

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
        coords = repeat(coords, 'n coord -> b n coord h w', b = B, h = H, w = W)

        patches = torch.tensor(patches)
        patches_with_coords = torch.cat([patches, coords], dim = 2)

        return patches_with_coords