from json import load
from tkinter import BaseWidget
import pytorch_lightning as pl
import torch
from torch.mps import current_allocated_memory
from torch.nn.utils.spectral_norm import SpectralNormLoadStateDictPreHook
import wandb as wb
import math

from torch.nn import functional as F
from torch.optim import AdamW, Adam
from torchvision.transforms import ToPILImage
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from torchmetrics import Accuracy

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
        self.warmup_backbone = 10
        self.router_start_epoch = 30
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

        # Best validation loss
        self.best_val_loss = float('+inf')

        self.aux_loss_weight = 0.0

        # Accuracy metrics
        self.accuracy_metrics = {
            'top1_train' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device),
            'top5_train' : Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device),

            'top1_val' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device),
            'top5_val' : Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
        }
        # Loss function
        self.val_loss = torch.nn.CrossEntropyLoss()
        self.train_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.05)


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

        logits, z_loss, imb_loss = self(data)

        class_loss = self.train_loss(logits, labels)
        z_loss_weigth = 1e-6 if self.current_epoch >= self.router_start_epoch else 1e-10

        aux_loss = (imb_loss * self.aux_loss_weight) + (z_loss * z_loss_weigth)
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

        torch.nn.utils.clip_grad_norm_(self.router_params, max_norm=0.1)
        torch.nn.utils.clip_grad_norm_(self.backbone_params, max_norm=1.0)

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
            k = float(self.k_scheduler)
            for l in self.model.layers:
                l.router_gate.k = k

    #----- SCHEDULERS -----
    def temp_scheduler(self):
        t0 = self.router_start_epoch
        tw = self.router_warmup

        tdec = max(self.temp_epochs, t0  + tw + 1)

        e = self.current_epoch

        # During warmup and uniform phases
        if e < t0 or e < t0 + tw:
            return self.temp_init
        
        # Cosine decay
        if e < tdec:
            progress = (self.current_epoch - (t0 + tw)) / float(max(1, tdec - (t0 + tw)))
            progress = min(max(progress, 0.0), 1.0)
            cosine_inc = 0.5 * (1.0 - math.cos(math.pi * progress))
            return self.temp_init + (self.temp_final - self.temp_init) * cosine_inc
        
        return self.temp_final

    def alpha_scheduler(self):
        e = self.current_epoch
        alpha_init  = self.alpha_init
        alpha_final = self.alpha_final

        t0 = self.router_start_epoch          
        tw = self.router_warmup         
        te = self.alpha_epochs

        if e < t0:
            return 0.0
        

        alpha_init  = float(self.alpha_init)
        alpha_final = float(self.alpha_final)

        alpha_warm_end = alpha_init
        if e < t0 + tw:
            pct = (e - t0 + 1) / float(max(1, tw))  # 0→1
            return alpha_init + (alpha_warm_end - alpha_init) * pct
        
        tdec = t0 + tw + int(self.alpha_epochs)
        
        if e < tdec:
            progress = (e - (t0 + tw)) / float(max(1, tdec - (t0 + tw)))  # 0→1
            progress = min(max(progress, 0.0), 1.0)
            cosine_inc = 0.5 * (1.0 - math.cos(math.pi * progress))      # 0→1
            return alpha_warm_end + (alpha_final - alpha_warm_end) * cosine_inc

        return alpha_final
    
    def k_scheduler(self):
        e = self.current_epoch
        t0 = self.router_start_epoch
        tw = self.router_warmup

        tdec = max(self.temp_epochs, t0 + tw + 1)
        
        if e < t0 or e < t0 + tw:
            return 1.0
        
        if e < tdec:
            progress = (e - (t0 + tw)) / float(max(1, tdec - (t0 + tw)))
            progress = min(max(progress, 0.0), 1.0)

            slow_progress = progress ** 2

            k_final = 0.25 
            return 1.0 - slow_progress * (1.0 - k_final)

        return 0.25
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
            'Gradient norm router' : torch.tensor(self.gradient_norm_router).mean().item()
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
        
        logits, z_loss, imb_loss = self(data)

        class_loss = self.val_loss(logits, labels)
        z_loss_weigth = 1e-4 if self.current_epoch >= self.router_start_epoch else 1e-10

        aux_loss = (imb_loss * self.aux_loss_weight) + (z_loss * z_loss_weigth)
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
        wd = self.weight_decay

        tot_epochs = self.train_epochs
        warmup_backbone = self.warmup_backbone
        router_start_epoch = self.router_start_epoch
        router_warmup = self.router_warmup

        eta_min = 1e-6

        def backbone_lr_lambda(epoch):
            if epoch < warmup_backbone:
                pct = epoch / warmup_backbone
                return eta_min + (1 - eta_min) * pct
            else : 
                progress = (epoch - warmup_backbone) / (tot_epochs - warmup_backbone) 
                progress = min(progress, 1.0)

                cosine_dec = 0.5 * (1 + math.cos(math.pi * progress))
                return eta_min + (1 - eta_min) * cosine_dec

        def router_lr_lambda(epoch):
            if epoch < router_start_epoch:
               return 0.0

            # Linear Warmup
            if epoch < router_start_epoch + router_warmup:
                pct = (epoch - router_start_epoch + 1) / float(max(1, router_warmup))
                return eta_min + (1 - eta_min) * pct

            # Cosine decay
            start = router_start_epoch + router_warmup
            progress = (epoch - start) / float(max(1, tot_epochs - start))
            progress = min(progress, 1.0)
            cosine_dec = 0.5 * (1 + math.cos(math.pi * progress))
            return eta_min + (1 - eta_min) * cosine_dec

        # Optimizer
        self.optimizer = AdamW(
            [
                {'params': self.backbone_params, 'lr': base_lr, 'weight_decay': wd, 'name' : 'backbone'},
                {'params': self.router_w, 'lr': base_lr * self.router_mul, 'weight_decay': 0, 'name' : 'router_w'},
                {'params' : self.router_bias, 'lr' : base_lr * self.router_mul, 'weight_decay': 0, 'name' : 'router_bias'}
            ], betas=(0.9, 0.999)
        )

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=[
            backbone_lr_lambda,
            router_lr_lambda,
            router_lr_lambda
        ])

        return [self.optimizer], [self.lr_scheduler]

    def _freeze_router(self):
        for l in self.model.layers:
            for p in l.router_gate.parameters():
                p.requires_grad_(False)

        for p in self.model.router.parameters():
            p.requires_grad_(False)
    
    def _unfreeze_router(self):
        for l in self.model.layers:
            for p in l.router_gate.parameters():
                p.requires_grad_(True)
        
        for p in self.model.router.parameters():
            p.requires_grad_(True)

    def _freeze_backbone(self):
        for name, p in self.model.named_parameters():
            if "router" in name or "prediction_head" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def _unfreeze_all(self):
        for _, p in self.model.named_parameters():
            p.requires_grad = True