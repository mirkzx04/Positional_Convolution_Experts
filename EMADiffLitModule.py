from json import load
from tkinter import BaseWidget
import pytorch_lightning as pl
import torch
from torch.mps import current_allocated_memory
from torch.nn.utils.spectral_norm import SpectralNormLoadStateDictPreHook
import wandb as wb
import math
import random

from torch.nn import functional as F
from torch.optim import AdamW, Adam
from torchvision.transforms import ToPILImage
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from torchmetrics import Accuracy

import torch.nn as nn

from Model.Components.DownsampleResBlock import DownsampleResBlock

class EMADiffLitModule(pl.LightningModule):
    def __init__(
                self, 
                pce,
                lr, 
                weight_decay,
                num_classes,
                device,
                train_epochs,
                uniform_epochs,
                alpha_init,
                alpha_final,
                alpha_epochs,
                temp_init,
                temp_mid,
                temp_final,
                temp_epochs,
            ):
        
        """
        Initialize the EMADiffLitModule.

        Args:
            num_experts (int): Number of experts.
            layer_number (int): Number of layers.
            patch_size (int): Patch size.
            dropout (float): Dropout rate.
            num_classes (int): Number of classes.
            nucleus_sampling_p (float): Nucleus sampling probability.
            lr (float): Learning rate.
            weight_decay (float): Weight decay.
            augmentation: Data augmentation object.
            class_names (list): List of class names.
            device (str): Device to use.
        Returns:
            None
        """

        super().__init__()
        # Model and optimizer parameters
        self.model = pce
        self.lr = lr
        self.weight_decay = weight_decay

        # Router hyperparameters for scheduling
        self.alpha_init = alpha_init # Load balancinc wheigth
        self.alpha_final = alpha_final
        self.alpha_epochs = alpha_epochs

        self.temp_init = temp_init # Logits router temperature
        self.temp_mid = temp_mid
        self.temp_final = temp_final
        self.temp_epochs = temp_epochs
        self.num_classes = num_classes

        self.router_mul = 2.0
        self.warmup_backbone = 15
        self.router_start_epoch = uniform_epochs
        self.router_warmup = 5
        self.use_augmentation = True
        
        self.train_epochs = train_epochs
        self.uniform_epochs = uniform_epochs

        # Training losses
        self.train_class_losses = []
        self.train_aux_losses = []
        self.train_total_losses = []

        # Validation losses
        self.val_class_losses = []
        self.val_aux_losses = []
        self.val_total_losses = []

        self.gradient_norm_router = []
        self.gradient_norm_backbone = []

        # Best validation loss
        self.best_val_loss = float('+inf')

        self.aux_loss_weight = 0.0

        self.use_mixup_cutmix = True
        self.mixup_alpha      = 0.2   
        self.cutmix_alpha     = 0.5
        self.cutmix_prob      = 0.3

        # Accuracy metrics
        self.accuracy_metrics = {
            'top1_train' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device),
            'top5_train' : Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device),

            'top1_val' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device),
            'top5_val' : Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
        }
        # Loss function
        self.val_loss = torch.nn.CrossEntropyLoss()
        self.train_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)


    def forward(self, x, force_specialized = False):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits from the model.
        """
        return self.model(x, current_epoch=self.current_epoch,)
    
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): Batch of data (inputs, labels).
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing predictions, losses, and batch index.
        """
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)

        if self.use_mixup_cutmix:
            r = random.random()
            if r < self.cutmix_prob:
                # CutMix
                data, targets_a, targets_b, lam = self._cutmix_batch(data, labels)
            else:
                # Mixup
                data, targets_a, targets_b, lam = self._mixup_batch(data, labels)

            logits, z_loss, imb_loss, div_loss = self(data)

            # Loss = lam * CE(logits, y_a) + (1-lam) * CE(logits, y_b)
            class_loss = (
                lam * self.train_loss(logits, targets_a)
                + (1.0 - lam) * self.train_loss(logits, targets_b)
            )

        else:
            logits, z_loss, imb_loss, div_loss = self(data)
            class_loss = self.train_loss(logits, labels)
        
        z_loss_weigth = 1e-3 if self.current_epoch >= self.router_start_epoch else 0
        div_loss_w = 1e-3

        aux_loss = (imb_loss * self.aux_loss_weight) + (z_loss * z_loss_weigth) + (div_loss_w * div_loss) 
        total_loss = class_loss + aux_loss

        self.train_class_losses.append(class_loss.item())
        self.train_aux_losses.append(aux_loss.item())
        self.train_total_losses.append(total_loss.item())
        
        if self.num_classes >= 5:
            if self.use_augmentation and labels.dim() > 1 :
                labels_accuracy = torch.argmax(labels, dim = 1)
            else : 
                labels_accuracy = labels

            self.accuracy_metrics['top1_train'].update(logits, labels_accuracy)
            self.accuracy_metrics['top5_train'].update(logits, labels_accuracy)

        return {'loss' : total_loss}

    def on_after_backward(self):
        """
        Calculate the gradient norm after backward pass.

        Returns:
            None
        """
        total_norm_router = total_norm_backbone = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                if 'router' in name or 'gate' in name or 'router_gate' in name:
                    param_norm_router = p.grad.data.norm(2)
                    total_norm_router += param_norm_router.item() ** 2
                else:
                    param_norm_backbone = p.grad.data.norm(2)
                    total_norm_backbone += param_norm_backbone.item()**2
        
        self.gradient_norm_router.append(total_norm_router ** 0.5)
        self.gradient_norm_backbone.append(total_norm_backbone ** 0.5)

        torch.nn.utils.clip_grad_norm_(self.router_params, max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.backbone_params, max_norm=1.5)

    def on_train_epoch_start(self):
        self.accuracy_metrics['top1_train'].reset()
        self.accuracy_metrics['top5_train'].reset()

        with torch.no_grad():
            self.model.moe_aggregator.reset()
            

        e = self.current_epoch
        if e < self.uniform_epochs:
            self.aux_loss_weight = 0
            self._freeze_router()

        elif e >= self.router_start_epoch:
            self._unfreeze_router()
            self.aux_loss_weight = self.alpha_scheduler()
            self.model.router.router_temp = self.temp_scheduler()
            self.model.router.noise_std = self.noise_scheduler()

    #----- SCHEDULERS -----
    def temp_scheduler(self):
        e  = int(self.current_epoch)
        t0 = int(self.router_start_epoch)
        tw = int(self.router_warmup)
        T  = int(self.train_epochs)

        t_start = t0 + tw
        t_end   = min(t_start + int(self.temp_epochs), T)

        if e < t_start:
            return float(self.temp_init)

        if e < t_end:
            progress   = (e - t_start) / float(max(1, t_end - t_start))
            progress   = min(max(progress, 0.0), 1.0)
            cosine_dec = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
            return float(self.temp_final) + (float(self.temp_init) - float(self.temp_final)) * cosine_dec

        return float(self.temp_final)

    def alpha_scheduler(self):
        e  = int(self.current_epoch)
        rs = int(self.router_start_epoch)
        rw = int(self.router_warmup)
        T  = int(self.train_epochs)

        a_peak  = float(self.alpha_init)
        a_final = float(self.alpha_final)

        # Router non attivo: nessuna aux loss
        if e < rs:
            return 0.0

        # Warmup: 0 -> a_peak
        if e < rs + rw:
            pct = (e - rs + 1) / float(max(1, rw))
            return a_peak * pct

        # Plateau a a_peak finché il router esplora a LR pieno
        t1_alpha = rs + int(0.4 * (T - rs))  # ~40% della parte post-router
        if e < t1_alpha:
            return a_peak

        # Decadimento coseno a_peak -> a_final da t1_alpha a T
        progress   = (e - t1_alpha) / float(max(1, T - t1_alpha))
        progress   = min(max(progress, 0.0), 1.0)
        cosine_dec = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return a_final + (a_peak - a_final) * cosine_dec

    def noise_scheduler(self):
        epoch = int(self.current_epoch)
        rs    = int(self.router_start_epoch)
        T     = int(self.train_epochs)

        noise_max = 0.05
        noise_min = 0.03

        # Prima che il router sia attivo: niente rumore
        if epoch < rs:
            return 0.0

        # Plateau di esplorazione ad alto rumore
        t1_noise = rs + int(0.3 * (T - rs))  # ~30% della parte post-router
        if epoch < t1_noise:
            return noise_max

        # Decadimento coseno noise_max -> noise_min
        progress   = (epoch - t1_noise) / float(max(1, T - t1_noise))
        progress   = min(max(progress, 0.0), 1.0)
        cosine_dec = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return noise_min + (noise_max - noise_min) * cosine_dec
    #----- SCHEDULERS -----

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch to compute and reset average losses.

        Returns:
            None
        """

        # Log dictionary
        log_dict = {
            'training/train_class_loss' : torch.tensor(self.train_class_losses).mean().item(),
            'training/train_aux_loss' : torch.tensor(self.train_aux_losses).mean().item(),
            'training/train_total_loss' : torch.tensor(self.train_total_losses).mean().item(),
            'training/train_top1' : self.accuracy_metrics['top1_train'].compute().item() * 100,
            'training/train_top5' : self.accuracy_metrics['top5_train'].compute().item() * 100,
            'LR_backbone : ' : self.optimizer.param_groups[0]['lr'],
            'LR_Router' : self.optimizer.param_groups[1]['lr'],
            'Gradient norm backbone' : torch.tensor(self.gradient_norm_backbone).mean().item(),
            'Gradient norm router' : torch.tensor(self.gradient_norm_router).mean().item(),
            'alpha_loss' : torch.tensor(self.aux_loss_weight),
            'temp_logits' : torch.tensor(self.model.router.router_temp),
        }

        self.train_class_losses.clear()
        self.train_aux_losses.clear()
        self.train_total_losses.clear()
        self.gradient_norm_router.clear()
        self.gradient_norm_backbone.clear()

        with torch.no_grad():
            router_metrics = self.model.moe_aggregator.finalize()
        log_dict.update({f'router-train/{k}' : v for k, v in router_metrics.items()})

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (tuple): Batch of data (inputs, labels).
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing predictions, losses, batch index, and loss histories.
        """ 
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)
        
        logits, z_loss, imb_loss, div_loss = self(data)

        class_loss = self.val_loss(logits, labels)

        z_loss_weigth = 1e-3 if self.current_epoch >= self.router_start_epoch else 1e-10
        div_loss_w = 1e-3

        aux_loss = (imb_loss * self.aux_loss_weight) + (z_loss * z_loss_weigth) + (div_loss_w * div_loss) 
        total_loss = class_loss + aux_loss   

        self.val_class_losses.append(class_loss.item())
        self.val_aux_losses.append(aux_loss.item())
        self.val_total_losses.append(total_loss.item())

        if self.num_classes >= 5:
            self.accuracy_metrics['top1_val'].update(logits, labels)
            self.accuracy_metrics['top5_val'].update(logits, labels)

        return {'loss' : total_loss}

    def on_validation_epoch_start(self):
        self.accuracy_metrics['top1_val'].reset()
        self.accuracy_metrics['top5_val'].reset()

        with torch.no_grad():
            self.model.moe_aggregator.reset()

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to compute and reset average losses.

        Returns:
            None
        """
        # Log dictionary
        log_dict = {
            'validation/val_class_loss' : torch.tensor(self.val_class_losses).mean().item(),
            'validation/val_aux_loss' : torch.tensor(self.val_aux_losses).mean().item(),
            'validation/val_total_loss' : torch.tensor(self.val_total_losses).mean().item(),
            'validation/val_top1' : self.accuracy_metrics['top1_val'].compute().item() * 100,
            'validation/val_top5' : self.accuracy_metrics['top5_val'].compute().item() * 100,
        }

        self.val_class_losses.clear()
        self.val_aux_losses.clear()
        self.val_total_losses.clear()

        with torch.no_grad():
            router_metrics = self.model.moe_aggregator.finalize()
        log_dict.update({f'router-val/{k}' : v for k, v in router_metrics.items()})

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        self.router_w, self.router_bias, self.backbone_params = [], [], []
        for name, p in self.model.named_parameters():
            if 'router_gate' in name or 'router' in name or 'gate' in name:
                (self.router_bias if p.dim()==1 else self.router_w).append(p)
            else:
                self.backbone_params.append(p)

        self.router_params = self.router_bias + self.router_w 

        base_lr = self.lr
        router_mul = self.router_mul
        wd = self.weight_decay

        tot_epochs       = self.train_epochs
        warmup_backbone  = self.warmup_backbone
        router_start_epoch = self.router_start_epoch
        router_warmup    = self.router_warmup

        eta_min         = 1e-3      
        eta_min_router  = 0.10
        router_min_factor = 0.30 

        def backbone_lr_lambda(epoch: int):
            e  = int(epoch)
            wb = int(warmup_backbone)
            rs = int(router_start_epoch)
            T  = int(tot_epochs)

            # Pivot: inizio del cosine decay del backbone
            t1_backbone = rs + int(0.2 * (T - rs))  # ~20% della parte post-router

            # 1) Warmup: eta_min -> 1.0
            if e < wb:
                pct = (e + 1) / float(max(1, wb))  # (0,1]
                return eta_min + (1.0 - eta_min) * pct

            # 2) Plateau LR pieno fino a t1_backbone
            if e < t1_backbone:
                return 1.0

            # 3) Cosine decay 1.0 -> eta_min da t1_backbone a T
            progress   = (e - t1_backbone) / float(max(1, T - t1_backbone))
            progress   = min(max(progress, 0.0), 1.0)
            cosine_dec = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
            return eta_min + (1.0 - eta_min) * cosine_dec

        def router_lr_lambda(epoch: int):
            e  = int(epoch)
            rs = int(router_start_epoch)
            rw = int(router_warmup)
            T  = int(tot_epochs)

            # Pivot per il router: plateau un po' più lungo del backbone
            t1_router = rs + int(0.35 * (T - rs))  # ~35% della parte post-router

            # Router spento prima di router_start_epoch
            if e < rs:
                return 0.0

            # Warmup router: eta_min_router -> 1.0
            if e < rs + rw:
                pct = (e - rs + 1) / float(max(1, rw))  # (0,1]
                return eta_min_router + (1.0 - eta_min_router) * pct

            # Plateau LR pieno
            if e < t1_router:
                return 1.0

            # Cosine decay 1.0 -> router_min_factor da t1_router a T
            progress   = (e - t1_router) / float(max(1, T - t1_router))  # 0..1
            progress   = min(max(progress, 0.0), 1.0)
            cosine_dec = 0.5 * (1.0 + math.cos(math.pi * progress))      # 1 -> 0
            return router_min_factor + (1.0 - router_min_factor) * cosine_dec

        # Optimizer 2e-5
        self.optimizer = AdamW(
            [
                {'params': self.backbone_params, 'lr': base_lr, 'weight_decay': wd, 'name' : 'backbone'},
                {'params': self.router_w, 'lr': 2e-5, 'weight_decay': 0, 'name' : 'router_w'},
                {'params' : self.router_bias, 'lr' : 2e-5, 'weight_decay': 0, 'name' : 'router_bias'}
            ], betas=(0.9, 0.98)
        )

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=[
            backbone_lr_lambda,
            router_lr_lambda,
            router_lr_lambda
        ])

        return [self.optimizer], [self.lr_scheduler]

    def _freeze_router(self):
        for l in self.model.layers:
            if not isinstance(l, DownsampleResBlock):
                for p in l.router_gate.parameters():
                    p.requires_grad_(False)

        for p in self.model.router.parameters():
            p.requires_grad_(False)
    
    def _unfreeze_router(self):
        for l in self.model.layers:
            if not isinstance(l, DownsampleResBlock):
                for p in l.router_gate.parameters():
                    p.requires_grad_(True)
        
        for p in self.model.router.parameters():
            p.requires_grad_(True)

    # MIXUP - CutMix utils
    def _sample_lambda(self, alpha) : 
        """
        Sample lambda from beta
        """
        if alpha <= 0:
            return 1.0
        
        beta_dist = torch.distributions.Beta(alpha, alpha)
        lam = beta_dist.sample().item()
        return float(lam)
    
    def _mixup_batch(self, x, y):
        """
        applly Mixup on batch
        """
        if self.mixup_alpha <= 0:
            return x,y,y,1.0
        
        lam = self._sample_lambda(self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1.0 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def _rand_box(self, size, lam):
        """
        Compute a random box for CutMix
        size : (B, C, H, W)
        """

        B, C, H, W = size
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Center of box
        cx = torch.randint(W, (1,), device=self.device).item()
        cy = torch.randint(H, (1,), device=self.device).item()

        bbx1 = max(cx - cut_w // 2, 0)
        bby1 = max(cy - cut_h // 2, 0)
        bbx2 = min(cx + cut_w // 2, W)
        bby2 = min(cy + cut_h // 2, H)

        return bbx1, bby1, bbx2, bby2
    
    def _cutmix_batch(self, x, y):
        """
        Apply cutmix on batch
        """
        if self.cutmix_alpha <= 0.0:
            return x, y, y, 1.0

        lam = self._sample_lambda(self.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = self._rand_box(x.size(), lam)

        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        cut_area = (bbx2 - bbx1) * (bby2 - bby1)
        lam = 1.0 - cut_area / float(x.size(2) * x.size(3))

        return x, y_a, y_b, lam