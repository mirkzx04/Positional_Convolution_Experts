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
                router_lr,
                weight_decay,
                num_classes,
                device,
                train_epochs,
                uniform_epochs,
                temp_init,
                temp_mid,
                temp_final,
                temp_epochs,
                covarage_init,
                covarage_final,
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
        self.router_lr = router_lr
        self.weight_decay = weight_decay

        self.temp_init = temp_init # Logits router temperature
        self.temp_mid = temp_mid
        self.temp_final = temp_final
        self.temp_epochs = temp_epochs
        self.num_classes = num_classes

        self.covarage_init = covarage_init
        self.covarage_final = covarage_final

        self.router_mul = 2.0
        self.warmup_backbone = 15
        self.router_start_epoch = uniform_epochs
        self.router_warmup = 10
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
        self.last_conv_grad_norm_sum_layers = []
        self.last_conv_grad_norm_count_layers = []

        # Best validation loss
        self.best_val_loss = float('+inf')

        self.use_mixup_cutmix = True
        self.mixup_alpha      = 0.2   
        self.cutmix_alpha     = 1.0
        self.cutmix_prob      = 0.3

        self.div_loss_weight = 0.03
        self.z_loss_weigth = 0.0
        self.cov_loss_weight = 0.0

        # Accuracy metrics
        self.accuracy_metrics = {
            'top1_train' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device),
            'top5_train' : Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device),

            'top1_val' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device),
            'top5_val' : Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
        }
        # Loss function
        self.val_loss = torch.nn.CrossEntropyLoss()
        self.train_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.10) # M
        self._reset_last_conv_grad_metrics()


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

            logits, cov_loss, z_loss, div_loss = self(data)

            # Loss = lam * CE(logits, y_a) + (1-lam) * CE(logits, y_b)
            class_loss = (
                lam * self.train_loss(logits, targets_a)
                + (1.0 - lam) * self.train_loss(logits, targets_b)
            )

        else:
            logits, cov_loss, z_loss, div_loss = self(data)
            class_loss = self.train_loss(logits, labels)
        
        aux_loss = (cov_loss * self.cov_loss_weight) + (z_loss * self.z_loss_weigth) + (self.div_loss_weight * div_loss) 
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
        self._update_last_conv_grad_metrics()

        torch.nn.utils.clip_grad_norm_(self.router_params, max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.backbone_params, max_norm=1.5)

    def on_train_epoch_start(self):
        self.accuracy_metrics['top1_train'].reset()
        self.accuracy_metrics['top5_train'].reset()

        with torch.no_grad():
            self.model.moe_aggregator.reset()
        self._reset_last_conv_grad_metrics()
            

        e = self.current_epoch
        if e < self.uniform_epochs:
            self._freeze_router()

        elif e >= self.router_start_epoch:
            self._unfreeze_router()
            self.z_loss_weigth = 1e-3
            self.cov_loss_weight = self.covarage_scheduler()
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

    def noise_scheduler(self):
        e = int(self.current_epoch)
        rs    = int(self.router_start_epoch)
        rw = int(self.router_warmup)
        T     = int(self.train_epochs)

        t_start = rs + rw 
        t_end = min(t_start + int(self.temp_epochs), T)

        noise_max = 0.02
        noise_final = 0.0

        # Prima che il router sia attivo: niente rumore
        if e < rs:
            return 0.0

        # Plateau di esplorazione ad alto rumore
        if e < t_start:
            return noise_max

        if e < t_end :
            progress = (e - t_start) / float(max(1, t_end - t_start))
            progress = min(max(progress, 0.0), 1.0)
            cosine_dec = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
            return noise_final + (noise_max - noise_final) * cosine_dec
        
        return noise_final


    def covarage_scheduler(self):
        e = int(self.current_epoch)
        rs = int(self.router_start_epoch)
        rw = int(self.router_warmup)
        T = int(self.train_epochs)

        c_init = float(self.covarage_init)   # es. 1e-2
        c_final = float(self.covarage_final) # es. 1e-3

        if e < rs:
            return 0.0

        warmup_end = min(rs + rw, T)
        actual_warmup = max(1, warmup_end - rs)

        # warmup: 0 -> c_init
        if e < warmup_end:
            progress = (e - rs + 1) / float(actual_warmup)
            progress = min(max(progress, 0.0), 1.0)
            return c_init * progress

        if warmup_end >= T - 1:
            return c_final

        # decay: c_init -> c_final
        decay_steps = max(1, T - warmup_end - 1)
        progress = (e - warmup_end) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)

        cosine_dec = 0.5 * (1.0 + math.cos(math.pi * progress))
        return c_final + (c_init - c_final) * cosine_dec
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
            'LR_Router' : self.optimizer.param_groups[2]['lr'],
            'Gradient norm backbone' : torch.tensor(self.gradient_norm_backbone).mean().item(),
            'Gradient norm router' : torch.tensor(self.gradient_norm_router).mean().item(),
            'temp_logits' : torch.tensor(self.model.router.router_temp),
        }

        self.train_class_losses.clear()
        self.train_aux_losses.clear()
        self.train_total_losses.clear()
        self.gradient_norm_router.clear()
        self.gradient_norm_backbone.clear()

        log_dict.update(self._collect_router_metrics('router-train'))
        log_dict.update(self._collect_last_conv_metrics('router-train'))

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
        
        logits, cov_loss, z_loss, div_loss = self(data)

        class_loss = self.val_loss(logits, labels)

        aux_loss = (cov_loss * self.cov_loss_weight) + (z_loss * self.z_loss_weigth) + (self.div_loss_weight * div_loss) 
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

        log_dict.update(self._collect_router_metrics('router-val'))

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def _collect_router_metrics(self, prefix: str):
        with torch.no_grad():
            router_metrics = self.model.moe_aggregator.finalize()

        log_dict = {}
        for key, value in router_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            log_dict[f'{prefix}/{key}'] = float(value)
        return log_dict

    def _iter_moe_layers(self):
        moe_layer_idx = 0
        for layer in self.model.layers:
            if isinstance(layer, DownsampleResBlock):
                continue
            yield moe_layer_idx, layer
            moe_layer_idx += 1

    def _reset_last_conv_grad_metrics(self):
        num_moe_layers = sum(
            0 if isinstance(layer, DownsampleResBlock) else 1
            for layer in self.model.layers
        )
        self.last_conv_grad_norm_sum_layers = [0.0 for _ in range(num_moe_layers)]
        self.last_conv_grad_norm_count_layers = [0 for _ in range(num_moe_layers)]

    def _update_last_conv_grad_metrics(self):
        for moe_layer_idx, layer in self._iter_moe_layers():
            grad_norms = []
            for expert in layer.experts:
                grad = expert.last_conv.weight.grad
                grad_norms.append(0.0 if grad is None else float(grad.detach().norm(2).item()))

            if grad_norms:
                self.last_conv_grad_norm_sum_layers[moe_layer_idx] += (
                    sum(grad_norms) / len(grad_norms)
                )
                self.last_conv_grad_norm_count_layers[moe_layer_idx] += 1

    def _collect_last_conv_metrics(self, prefix: str):
        log_dict = {}

        with torch.no_grad():
            for moe_layer_idx, layer in self._iter_moe_layers():
                weight_norms = [
                    float(expert.last_conv.weight.detach().norm(2).item())
                    for expert in layer.experts
                ]
                mean_weight_norm = sum(weight_norms) / max(1, len(weight_norms))
                log_dict[
                    f'{prefix}/moe_layer_{moe_layer_idx}/last_conv_weight_norm'
                ] = float(mean_weight_norm)

                grad_steps = self.last_conv_grad_norm_count_layers[moe_layer_idx]
                mean_grad_norm = (
                    self.last_conv_grad_norm_sum_layers[moe_layer_idx] / grad_steps
                    if grad_steps > 0 else 0.0
                )
                log_dict[
                    f'{prefix}/moe_layer_{moe_layer_idx}/last_conv_grad_norm'
                ] = float(mean_grad_norm)

        return log_dict

    def configure_optimizers(self):

        backbone_params, backbone_norm = [], []
        router_params = []

        for n, p in self.model.named_parameters():
            is_router = ('router_gate' in n) or ('router' in n) or ('gate' in n)

            if  n.endswith('.bias') or 'norm' in n.lower() or 'bn' in n.lower():
                if is_router:
                    router_params.append(p)
                else:
                    backbone_norm.append(p)
            else:
                if is_router : 
                    router_params.append(p)
                else:
                    backbone_params.append(p)

        self.backbone_params = backbone_params + backbone_norm
        self.router_params = router_params 

        base_lr = self.lr
        router_lr = self.router_lr
        wd = self.weight_decay

        T = self.train_epochs
        warmup_backbone = self.warmup_backbone

        router_start_epoch = self.router_start_epoch
        warmup_router = self.router_warmup

        def backbone_lr_lambda(epoch: int):
            e = int(epoch)
            if  e < warmup_backbone:
            # warmup 0 -> 1
                return (e + 1) / float(max(1, warmup_backbone))

            eta_min = 0.1
            
            # cosine decay da warmup_epochs a T
            progress = (e - warmup_backbone) / float(max(1, T - warmup_backbone))
            progress = min(max(progress, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
            return eta_min + (1.0 - eta_min) * cos            # 1 -> eta_min0

        def router_lr_lambda(epoch: int):
            e = int(epoch)
            # Router spento fino a router_start_epoch
            if e < router_start_epoch:
                return 0.0

            # Warmup router: 0 -> 1 in router_warmup epoche
            if e < router_start_epoch + warmup_router:
                pct = (e - router_start_epoch + 1) / float(max(1, warmup_router))
                return pct

            # Cosine decay da (router_start_epoch + router_warmup) a T
            eta_min_r = 0.10  # router spesso ok un po' più alto

            progress = (e - router_start_epoch) / float(max(1, T - router_start_epoch))
            progress = min(max(progress, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_r + (1.0 - eta_min_r) * cos

        # Optimizer 2e-5
        self.optimizer = AdamW(
            [
                {'params': backbone_params, 'lr': base_lr, 'weight_decay': wd, 'name' : 'backbone'}, # Conv and Linear
                {'params' : backbone_norm, 'lr': base_lr, 'weight_decay': 0, 'name' : 'backbone_norm'},
                {'params': router_params, 'lr': router_lr, 'weight_decay': 0, 'name' : 'router_w'},
            ], betas=(0.9, 0.98), eps=1e-8
        )

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=[
            backbone_lr_lambda,
            backbone_lr_lambda,
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
