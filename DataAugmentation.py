import torch
import torch.nn.functional as F
import numpy as np
import random
import math

class DataAgumentation:
    def __init__(self, augment_prob = 0.5, rotation_angles = [90,180,270], patch_size_range = (0.1, 0.3), shuffle_zones = 4):
        """
        Classes for data augmentation that applied casual transforms at batch

        Args:
            augment_prob : probabilities that one image in to batch will be transforms
            ration_angles : List of angle of the pissible rotation
            patch_size_range : range of patch dimension (in fraction), this patch will applied on one random zone of image (min. max)
            shuffle_zones : Number of zones where image comes  for pixel shuffling
        """        

        self.augment_prob = augment_prob
        self.rotation_angles = rotation_angles
        self.patch_size_range = patch_size_range
        self.shuffle_zones = shuffle_zones

        # Transforms list
        self.augmentations = [
            self.rotate_img,
            self.shuffle_pixels,
            self.add_random_patch
        ]

    def __call__(self, batch):
        """
        Applied data augmentation on random batch

        Args:
            batch : tensor [B, C, H, W]

        returns : 
            batch_augmented : batch with some image transforms
        """

        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)

        batch_augmented = batch.clone()
        batch_size = batch.shape[0]

        for i in range(batch_size):
            # Decide if applied augmentation of this image
            if random.random() < self.augment_prob:
                # Decide random transformrs
                aumentation_transform = random.choice(self.augmentations)
                batch_augmented[i] = aumentation_transform(batch[i])

        return batch_augmented
    
    def rotate_img(self, image):
        """
        Rotate image
        """
        
        angle = random.choice(self.rotation_angles)

        # Convert angle in radiant
        angle_rad = angle * math.pi / 180.0

        # Create rotation matrix
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Create theta matric [H, W]
        theta = torch.tensor([[cos_a, -sin_a, 0],
                              [sin_a, cos_a, 0]], dtype=torch.float32, device=image.device).unsqueeze(0)

        # Create grill and applied transform
        grid = F.affine_grid(theta, image.unsqueeze(0).shape, align_corners=False)
        rotated = F.grid_sample(image.unsqueeze(0), grid, align_corners=False)
        rotated = rotated.squeeze(0)

        return rotated
    
    def shuffle_pixels(self, image):
        """
        Shuffle pixel in random zone of the image
        """
        C, H, W = image.shape
        augmented = image.clone()

        # Divide image in zone
        zone_h = H // self.shuffle_zones
        zone_w = W // self.shuffle_zones

        # Choose some random zone for shuffling
        num_zones_to_shuffle = random.randint(1, self.shuffle_zones)

        for _ in range(num_zones_to_shuffle):
            # Choose a random zone
            start_h = random.randint(0, H - zone_h)
            start_w = random.randint(0, W - zone_w)

            end_h = start_h + zone_h
            end_w = start_w + zone_w

            # Extract the zone
            zone = augmented[:, start_h:end_h, start_w:end_w].clone()

            # Shuffle pixel in to zone
            zone_flat = zone.view(C, -1)
            indices = torch.randperm(zone_flat.shape[1])
            zone_shuffled = zone_flat[:, indices]

            augmented[:, start_h:end_h, start_w:end_w] = zone_shuffled.view(C, zone_h, zone_w)

        return augmented
    
    def add_random_patch(self, image):
        """
        Add a black or white patch in random position of image
        """
        C, H, W = image.shape
        augmented = image.clone()

        # Calculate dimension of patch
        min_size = int(min(H, W) * self.patch_size_range[0])
        max_size = int(min(H, W) * self.patch_size_range[1])
        patch_h = random.randint(min_size, max_size)
        patch_w = random.randint(min_size, max_size)

        # Place casual patch
        start_h = random.randint(0, H - patch_h)
        start_w = random.randint(0, W - patch_w)

        # Choose color : black (0) or white (1)
        patch_color = random.choice([0.0, 1.0])

        # Applied the patch
        augmented[:, start_h:start_h+patch_h, start_w:start_w+patch_w] = patch_color

        return augmented




