import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import random
import math
from typing import Optional, Tuple

class AdvancedDataAugmentation(nn.Module):
    def __init__(self, 
                 # RandAugment parameters
                 use_randaugment: bool = True,
                 randaugment_n: int = 2,
                 randaugment_m: int = 9,
                 
                 # Random Erasing
                 use_random_erasing: bool = True,
                 erasing_prob: float = 0.25,
                 erasing_scale: Tuple[float, float] = (0.02, 0.33),
                 erasing_ratio: Tuple[float, float] = (0.3, 3.3),
                 
                 # MixUp & CutMix
                 use_mixup: bool = True,
                 mixup_alpha: float = 0.2,
                 cutmix_alpha: float = 1.0,
                 cutmix_prob: float = 0.5,
                 label_smoothing: float = 0.1,
                 
                 # General
                 img_size: int = 32,
                 num_classes: int = 10):
        
        super().__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        self.use_randaugment = use_randaugment
        self.randaugment_n = randaugment_n
        self.randaugment_m = randaugment_m
        self.use_random_erasing = use_random_erasing
        self.erasing_prob = erasing_prob
        self.erasing_scale = erasing_scale
        self.erasing_ratio = erasing_ratio
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        self.label_smoothing = label_smoothing
        
        # CIFAR-10 normalization stats
        self.register_buffer('mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1))
        
        # Initialize torchvision transforms only - NO TIMM
        if self.use_randaugment:
            from torchvision.transforms import RandAugment
            self.randaugment = RandAugment(num_ops=randaugment_n, magnitude=randaugment_m)
        
        if self.use_random_erasing:
            from torchvision.transforms import RandomErasing
            self.random_erasing = RandomErasing(
                p=erasing_prob,
                scale=erasing_scale,
                ratio=erasing_ratio,
                value=0
            )
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to CIFAR-10 standard"""
        if x.max() > 1.5:
            x = x.float() / 255.0
        return (x - self.mean) / self.std
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor"""
        x = x * self.std + self.mean
        return torch.clamp(x, 0, 1)
    
    def apply_label_smoothing(self, y: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing"""
        if not self.training or self.label_smoothing <= 0:
            return y
        
        if y.dim() == 1:
            y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        else:
            y_onehot = y.float()
        
        smooth_labels = y_onehot * (1 - self.label_smoothing) + \
                       self.label_smoothing / self.num_classes
        return smooth_labels
    
    def apply_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp augmentation"""
        if not self.training or self.mixup_alpha <= 0:
            return x, y
        
        batch_size = x.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        indices = torch.randperm(batch_size, device=x.device)
        
        # Mix images
        mixed_x = lam * x + (1 - lam) * x[indices]
        
        # Mix labels with label smoothing
        y_smooth_a = self.apply_label_smoothing(y)
        y_smooth_b = self.apply_label_smoothing(y[indices])
        mixed_y = lam * y_smooth_a + (1 - lam) * y_smooth_b
        
        return mixed_x, mixed_y
    
    def apply_cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix augmentation"""
        if not self.training or self.cutmix_alpha <= 0:
            return x, y
        
        batch_size = x.size(0)
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        indices = torch.randperm(batch_size, device=x.device)
        
        # Generate random bbox
        _, _, h, w = x.shape
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        # Random center
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Calculate bbox
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        # Mix labels with label smoothing
        y_smooth_a = self.apply_label_smoothing(y)
        y_smooth_b = self.apply_label_smoothing(y[indices])
        mixed_y = lam * y_smooth_a + (1 - lam) * y_smooth_b
        
        return mixed_x, mixed_y
    
    def __call__(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, 
                augment_type: str = 'randaugment') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply data augmentation pipeline using torchvision only
        
        Args:
            x: Input images [B, C, H, W]
            y: Labels [B] (optional)
            augment_type: Type of augmentation ('randaugment', 'mixup', 'cutmix', 'labelsmooth', 'none')
            
        Returns:
            Augmented images and labels
        """
        # Normalize
        x = self.normalize(x)
        
        if not self.training or augment_type == 'none':
            return x, y
        
        # Apply RandAugment first (for all types except none)
        if self.use_randaugment and augment_type in ['randaugment', 'mixup', 'cutmix', 'labelsmooth']:
            batch_size = x.shape[0]
            augmented = []
            for i in range(batch_size):
                # Convert to PIL format for RandAugment
                img_tensor = self.denormalize(x[i])  # x[i] has shape [C, H, W]
                img_pil = T.ToPILImage()(img_tensor.cpu())
                # Apply RandAugment
                img_aug = self.randaugment(img_pil)
                # Convert back to tensor
                img_tensor = T.ToTensor()(img_aug).to(x.device)
                img_tensor = self.normalize(img_tensor)
                augmented.append(img_tensor)
            x = torch.stack(augmented)
        
        # Apply Random Erasing
        if self.use_random_erasing:
            x = self.random_erasing(x)
        
        # Apply augmentation based on type
        if y is not None:
            if augment_type == 'mixup':
                x, y = self.apply_mixup(x, y)
            elif augment_type == 'cutmix':
                if random.random() < self.cutmix_prob:
                    x, y = self.apply_cutmix(x, y)
                else:
                    # Fallback to label smoothing
                    y = self.apply_label_smoothing(y)
            elif augment_type in ['randaugment', 'labelsmooth']:
                y = self.apply_label_smoothing(y)
        
        return x, y
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, 
                augment_type: str = 'randaugment') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward method for nn.Module compatibility"""
        return self.__call__(x, y, augment_type)