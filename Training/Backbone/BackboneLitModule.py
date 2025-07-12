import pytotorch_lightning as pl
import torch

from torch.nn import functional as F
from torch.optim import Adam

from torchmetrics import Accuracy

from PCEScheduler import PCEScheduler

class BackboneLitModule(pl.LightningModule):
    def __init__(self, 
                model, 
                optimizer, 
                lr_scheduler,
                num_classes,
                start_epoch,
                start_train_batch,
                train_loss_history,
                val_loss_history,
                pre_train_epochs,
                fine_tune_epochs,
                phase_multipliers
            ):
        
        """
        Initialize the BackboneLitModule.

        Args:
            model (torch.nn.Module): The backbone model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            num_classes (int): Number of classes for classification tasks.
        """

        super().__init__()

        self.model = model
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.start_epoch = start_epoch
        self.start_train_batch = start_train_batch

        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history

        self.pre_train = pre_train_epochs
        self.fine_tune = fine_tune_epochs       
        self.lr_phase_multipliers = phase_multipliers

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_losses = []
        self.val_losses = []

        self.accuracy_metrics = {
            'top1_train' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1),
            'top5_train' : Accuracy(task='multiclass', num_classes=num_classes, top_k=5),

            'top1_val' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1),
            'top5_val' : Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        }

        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        """
        Forward pass through the model.
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): A tuple containing the input data and labels.
            batch_idx (int): The index of the current batch.
        """

        if self.current_epoch < self.start_epoch and batch_idx < self.start_train_batch:
            return None
        
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)

        logits = self(data)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        loss = self.criterion(logits, labels)
        self.train_loss_history.append(loss.item())

        self.train_losses.append(loss.item())

        if self.num_classes >= 5:
            self.accuracy_metrics['top1_train'].update(logits, labels)
            self.accuracy_metrics['top5_train'].update(logits, labels)

        return {
            'pred_labels': predictions,
            'loss': loss,
        }

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.
        """
        self.avg_train_loss = torch.tensor(self.train_losses).mean().item()
        self.train_losses.clear()

        self.gradient_norm = self.calculate_gradient_norm(self.model)

        return

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (tuple): A tuple containing the input data and labels.
            batch_idx (int): The index of the current batch.
        """
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)

        logits = self(data)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        loss = self.criterion(logits, labels)
        self.val_loss_history(loss.item())

        self.val_losses.append(loss.item())

        if self.num_classes >= 5:
            self.accuracy_metrics['top1_val'].update(logits, labels)
            self.accuracy_metrics['top5_val'].update(logits, labels)

        return {
            'pred_labels': predictions,
            'loss': loss,
        }

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch.
        """
        self.avg_val_loss = torch.tensor(self.val_losses).mean().item()
        self.val_losses.clear()

        return {
            'avg_val_loss': self.avg_val_loss,
            'avg_train_loss': self.avg_train_loss,
            'gradient_norm': self.gradient_norm,
            'train_top1_acc': self.accuracy_metrics['top1_train'].compute().item(),
            'train_top5_acc': self.accuracy_metrics['top5_train'].compute().item(),
            'val_top1_acc': self.accuracy_metrics['top1_val'].compute().item(),
            'val_top5_acc': self.accuracy_metrics['top5_val'].compute().item()
        }

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr = self.lr, weight_decay=self.weight_decay)
        lr_scheduler = PCEScheduler(
            optimizer= optimizer,
            phase_epochs=[self.pre_train, self.fine_tune],
            base_lr=self.lr,
            phase_multipliers=self.lr_phase_multipliers
        )

        if self.optimizer is not None:
            optimizer.load_state_dict(self.optimizer)
        if self.lr_scheduler is not None:
            lr_scheduler.load_state_dict(self.lr_scheduler)

        return {'optimizer' : optimizer, 'lr_scheduler' : lr_scheduler}

    def calculate_gradient_norm(model):
        """
        Calculate norm of gradient
        
        Args:
            model: pytorch model
            
        Returns:
            float: Global norm of gradient
        """
        total_grad_norm = 0.0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = torch.norm(param.grad.data).item()
                total_grad_norm += param_norm ** 2
        
        global_grad_norm = total_grad_norm ** 0.5
        return global_grad_norm
