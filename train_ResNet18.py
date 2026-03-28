import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import Accuracy
from torchvision.models import resnet18

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from main import get_tinyimagenet_sets

class ResNetLitModule(pl.LightningModule):
    def __init__(
                self, 
                lr, 
                weight_decay,
                num_classes,
                train_epochs,
                warmup_epochs=15,
                mixup_alpha=0.3,
                cutmix_alpha=1.0,
                cutmix_prob=0.5,
                label_smoothing=0.15
            ):
        """
        Lightning Module per ResNet-18 Standard.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = resnet18(weights = None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_epochs = train_epochs
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes

        # Mixup/CutMix params
        self.use_mixup_cutmix = True
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob

        # Metrics & Loss
        self.accuracy_metrics = nn.ModuleDict({
            'top1_train': Accuracy(task='multiclass', num_classes=num_classes, top_k=1),
            'top5_train': Accuracy(task='multiclass', num_classes=num_classes, top_k=5),
            'top1_val': Accuracy(task='multiclass', num_classes=num_classes, top_k=1),
            'top5_val': Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        })
        
        self.train_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.val_loss_fn = nn.CrossEntropyLoss()

        # Storage per logging medie
        self.train_losses = []
        self.val_losses = []
        self.backbone_grads = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        
        if self.use_mixup_cutmix:
            r = random.random()
            if r < self.cutmix_prob:
                data, targets_a, targets_b, lam = self._cutmix_batch(data, labels)
            else:
                data, targets_a, targets_b, lam = self._mixup_batch(data, labels)

            logits = self(data)
            loss = lam * self.train_loss_fn(logits, targets_a) + (1.0 - lam) * self.train_loss_fn(logits, targets_b)
        else:
            logits = self(data)
            loss = self.train_loss_fn(logits, labels)

        # Accuracy (usa labels originali o argmax se mixup)
        acc_labels = labels if labels.dim() == 1 else torch.argmax(labels, dim=1)
        self.accuracy_metrics['top1_train'].update(logits, acc_labels)
        self.accuracy_metrics['top5_train'].update(logits, acc_labels)
        
        self.train_losses.append(loss.item())
        return loss

    def on_after_backward(self):
        # Calcolo gradient norm semplificato solo per backbone
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.backbone_grads.append(total_norm ** 0.5)
        
        # Clip standard per ResNet
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def on_train_epoch_end(self):
        avg_loss = torch.tensor(self.train_losses).mean()
        avg_grad = torch.tensor(self.backbone_grads).mean()
        
        self.log_dict({
            'training/train_loss': avg_loss,
            'training/train_top1': self.accuracy_metrics['top1_train'].compute() * 100,
            'training/train_top5': self.accuracy_metrics['top5_train'].compute() * 100,
            'training/lr': self.trainer.optimizers[0].param_groups[0]['lr'],
            'training/grad_norm': avg_grad
        }, prog_bar=True)

        self.train_losses.clear()
        self.backbone_grads.clear()
        self.accuracy_metrics['top1_train'].reset()
        self.accuracy_metrics['top5_train'].reset()

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        logits = self(data)
        loss = self.val_loss_fn(logits, labels)

        self.val_losses.append(loss.item())
        self.accuracy_metrics['top1_val'].update(logits, labels)
        self.accuracy_metrics['top5_val'].update(logits, labels)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.tensor(self.val_losses).mean()
        self.log_dict({
            'validation/val_loss': avg_loss,
            'validation/val_top1': self.accuracy_metrics['top1_val'].compute() * 100,
            'validation/val_top5': self.accuracy_metrics['top5_val'].compute() * 100,
        }, prog_bar=True)
        
        self.val_losses.clear()
        self.accuracy_metrics['top1_val'].reset()
        self.accuracy_metrics['top5_val'].reset()

    def configure_optimizers(self):
        # Dividiamo i parametri per escludere Weight Decay da BN e bias
        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue
            if len(param.shape) == 1 or name.endswith(".bias") or "bn" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        
        optim_groups = [
            {'params': decay, 'weight_decay': self.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optim_groups, lr=self.lr, betas=(0.9, 0.98), eps=1e-8)

        # Scheduler originale riadattato (Warmup + Cosine Decay)
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / float(max(1, self.warmup_epochs))
            
            progress = (epoch - self.warmup_epochs) / float(max(1, self.train_epochs - self.warmup_epochs))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]

    # --- Utility Mixup/CutMix ---
    def _sample_lambda(self, alpha):
        if alpha <= 0: return 1.0
        return torch.distributions.Beta(alpha, alpha).sample().item()

    def _mixup_batch(self, x, y):
        lam = self._sample_lambda(self.mixup_alpha)
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1.0 - lam) * x[index, :]
        return mixed_x, y, y[index], lam

    def _cutmix_batch(self, x, y):
        lam = self._sample_lambda(self.cutmix_alpha)
        index = torch.randperm(x.size(0), device=x.device)
        
        # Coordinate box
        B, C, H, W = x.size()
        cut_rat = math.sqrt(1.0 - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = torch.randint(W, (1,)).item(), torch.randint(H, (1,)).item()

        bbx1, bby1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
        bbx2, bby2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)

        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Ricalcolo lam effettivo basato sull'area
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        return x, y, y[index], lam
    
if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_epochs = 150
    batch_size = 128
    lr = 5e-4
    wd = 1e-3

    tiny_set = get_tinyimagenet_sets(batch_size)
    train_loader = tiny_set['dataloaders']['train']
    val_loader = tiny_set['dataloaders']['val']
    num_classes = tiny_set['num_classes']

    lit = ResNetLitModule(
        lr=lr,
        weight_decay=wd,
        num_classes=num_classes,
        train_epochs=train_epochs
    )

    logger = WandbLogger(project="PCE", name="ResNet18-TinyImageNet-SameRecipe", log_model=False)
    logger.experiment.define_metric("epoch")
    logger.experiment.define_metric("*", step_metric="epoch")

    ckpt = ModelCheckpoint(
        dirpath="checkpoints",
        filename="resnet18_tiny_same_recipe",
        save_last=True,
        save_top_k=0,
        save_weights_only=False,
    )

    trainer = pl.Trainer(
        max_epochs=train_epochs,
        logger=logger,
        callbacks=[ckpt],
        precision="32",
        accelerator=DEVICE,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
    )

    trainer.fit(lit, train_loader, val_loader)
    logger.experiment.finish()