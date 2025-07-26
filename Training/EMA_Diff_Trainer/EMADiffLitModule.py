import pytorch_lightning as pl
import torch

from torch.nn import functional as F
from torch.optim import Adam

from torchmetrics import Accuracy

from PCEScheduler import PCEScheduler

class EMADiffLitModule(pl.LightningModule):
    def __init__(self, 
                PCE,
                lr_scheduler,
                optimizer,
                str_epoch,
                str_train_batch,
                str_val_batch,
                train_loss_history,
                val_loss_history,
                augmentation,
                phase_multipliers,
                lr,
                weight_decay,
                actual_phase,
                class_names,
                device
            ):
        
        """
        Initialize the EMADiffLitModule.

        Args:
            PCE (nn.Module): The PCE network to be trained.
            lr_scheduler: Learning rate scheduler.
            optimizer: Optimizer for training.
            str_epoch (int): Starting epoch (for resuming training).
            str_train_batch (int): Starting training batch (for resuming training).
            train_loss_history (list): List of training loss values.
            val_loss_history (list): List of validation loss values.
            augmentation: Data augmentation object.
            phase_multipliers (list): Multipliers for different training phases.
            lr (float): Learning rate.
            weight_decay (float): Weight decay for optimizer.
            actual_phase (str): Current training phase.
            num_classes (int): Number of classes for classification.
        Returns:
            None
        """

        super().__init__()

        self.model = PCE
        self.class_names = class_names
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.start_epoch = str_epoch
        self.start_train_batch = str_train_batch
        self.start_val_batch = str_val_batch

        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history
 
        self.lr_phase_multipliers = phase_multipliers
        self.lr = lr
        self.weight_decay = weight_decay

        self.augmentation = augmentation
        self.actual_phase = actual_phase

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_class_losses = []
        self.train_router_losses = []
        self.train_total_losses = []

        self.val_class_losses = []
        self.val_router_losses = []
        self.val_total_losses = []

        self.train_confidence_loss_history = []
        self.train_collapse_loss_history = []
        self.train_entropy_history = []

        self.val_confidence_loss_history = []
        self.val_collapse_loss_history = []
        self.val_entropy_history = []

        self.train_router_cache_history = []
        self.val_router_cache_history = []


        self.best_val_loss = float('+inf')

        self.accuracy_metrics = {
            'top1_train' : Accuracy(task='multiclass', num_classes=len(class_names), top_k=1).to(device),
            'top5_train' : Accuracy(task='multiclass', num_classes=len(class_names), top_k=5).to(device),

            'top1_val' : Accuracy(task='multiclass', num_classes=len(class_names), top_k=1).to(device),
            'top5_val' : Accuracy(task='multiclass', num_classes=len(class_names), top_k=5).to(device)
        }

        self.confidence_weight = 0.01
        self.anticollapse = 0.001

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits from the model.
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): Batch of data (inputs, labels).
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing predictions, losses, and batch index.
        """
        if self.current_epoch < self.start_epoch or batch_idx < self.start_train_batch:
            return None
        
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)
        data = self.augmentation(data)
        
        logits = self(data)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        class_loss = self.criterion(logits, labels)

        # Compute router loss
        router_cache = self.model.router.get_cache()
        self.train_router_cache_history.append(router_cache)
        router_loss = self.router_loss(router_cache)
        self.model.router.clear_cache()

        total_loss = class_loss + router_loss

        self.train_class_losses.append(class_loss.item())
        self.train_router_losses.append(router_loss.item())
        self.train_total_losses.append(total_loss.item())

        if len(self.class_names) >= 5:
            self.accuracy_metrics['top1_train'].update(logits, labels)
            self.accuracy_metrics['top5_train'].update(logits, labels)

        return {
            'pred_labels': predictions,
            'class_loss': class_loss,
            'router_loss': router_loss,
            'loss' : total_loss,
            'actual_batch' : batch_idx
        }

    def on_after_backward(self):
        self.gradient_norm = self.calculate_gradient_norm()
    
    def on_train_epoch_start(self):
        self.accuracy_metrics['top1_train'].reset()
        self.accuracy_metrics['top5_train'].reset()

    def my_on_train_epoch_end(self):
        """
        Called at the end of the training epoch to compute and reset average losses.

        Args:
            outputs: Not used.
            batch: Not used.
            batch_idx: Not used.

        Returns:
            dict: Dictionary with training and validation loss histories and current phase.
        """
        if self.start_train_batch != 0:
            self.start_train_batch = 0

        # Calc avg of training losses
        self.avg_train_class_losses = torch.tensor(self.train_class_losses).mean().item()
        self.avg_train_router_losses = torch.tensor(self.train_router_losses).mean().item()
        self.avg_train_total_losses = torch.tensor(self.train_total_losses).mean().item()

        self.train_class_losses.clear(), self.train_router_losses.clear(), self.train_total_losses.clear()

        # Calc router losses
        self.avg_train_confidence_loss = torch.tensor(self.train_confidence_loss_history).mean().item()
        self.avg_train_collapse_loss = torch.tensor(self.train_collapse_loss_history).mean().item()
        self.avg_train_patch_entropys = torch.cat(self.train_entropy_history).mean().item()

        self.train_confidence_loss_history.clear(), self.train_collapse_loss_history.clear(), self.train_entropy_history.clear()

        self.train_cache_stats = self.calc_cache_metrics()

        self.train_router_cache_history.clear()

        self.train_top1_acc = self.accuracy_metrics['top1_train'].compute().item() * 100
        self.train_top5_acc = self.accuracy_metrics['top5_train'].compute().item() * 100

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (tuple): Batch of data (inputs, labels).
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing predictions, losses, batch index, and loss histories.
        """ 
        if self.current_epoch < self.start_epoch and batch_idx < self.start_val_batch:
            return None
        
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)

        logits = self(data)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        class_loss = self.criterion(logits, labels)

        # Compute router loss
        router_cache = self.model.router.get_cache()
        self.router_cache_history.append(router_cache)
        router_loss = self.router_loss(router_cache)
        self.model.router.clear_cache()
        
        total_loss = class_loss + router_loss

        self.val_class_losses.append(class_loss.item())
        self.val_router_losses.append(router_loss.item())
        self.val_total_losses.append(total_loss.item())

        if len(self.class_names)>= 5:
            self.accuracy_metrics['top1_val'].update(logits, labels)
            self.accuracy_metrics['top5_val'].update(logits, labels)

        return {
            'pred_labels': predictions,
            'class_loss': class_loss,
            'router_loss': router_loss,
            'loss' : total_loss,
            'actual_batch' : batch_idx,
        }

    def on_validation_epoch_start(self):
        self.accuracy_metrics['top1_val'].reset()
        self.accuracy_metrics['top5_val'].reset()

    def my_on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to compute and reset average losses and metrics.

        Returns:
            dict: Dictionary with average losses, accuracies, epoch info, gradient norm, best validation loss, and router metrics.
        """
        if self.start_val_batch != 0:
            self.start_val_batch = 0

        self.avg_val_class_losses = torch.tensor(self.val_class_losses).mean().item()
        self.avg_val_router_losses = torch.tensor(self.val_router_losses).mean().item()
        self.avg_val_total_losses = torch.tensor(self.val_total_losses).mean().item()

        self.train_class_losses.clear(), self.train_router_losses.clear(), self.train_total_losses.clear()

        # Calc router losses
        self.avg_val_confidence_loss = torch.tensor(self.val_confidence_loss_history).mean().item()
        self.avg_val_collapse_loss = torch.tensor(self.val_collapse_loss_history).mean().item()
        self.avg_val_patch_entropys = torch.cat(self.val_entropy_history).mean().item()

        self.val_confidence_loss_history.clear(), self.val_collapse_loss_history.clear(), self.val_entropy_history.clear()

        self.val_cache_stats = self.calc_cache_metrics()

        self.val_router_cache_history.clear()

        self.val_top1_acc = self.accuracy_metrics['top1_val'].compute().item() * 100
        self.val_top5_acc = self.accuracy_metrics['top5_val'].compute().item() * 100

        if self.avg_val_total_losses < self.best_val_loss:
            self.best_val_loss = self.avg_val_total_losses

    def configure_optimizers(self):
        return [self.optimizer], [self.lr_scheduler]

    def calculate_gradient_norm(self):
        """
        Calculate norm of gradient
        
        Args:
            model: pytorch model
            
        Returns:
            float: Global norm of gradient
        """
        total_grad_norm = 0.0
        
        for param in self.model.parameters():
            if param.grad is not None and param.requires_grad:
                param_norm = torch.norm(param.grad.data).item()
                total_grad_norm += param_norm ** 2
        
        global_grad_norm = total_grad_norm ** 0.5
        return global_grad_norm
    
    def router_loss(self, router_cache):
        """
        Compute the router loss, including confidence and anti-collapse losses.

        Returns:
            Tensor or tuple: Router loss tensor, and optionally router metrics and cache metrics if self.router_metrics is True.
        """

        total_loss = 0.0

        if not router_cache:
            return torch.tensor(0.0, requires_grad=True)
        
        for layer_idx in range(len(router_cache)):
            # Get experts weights [B x P, num_experts]
            expert_weights = router_cache[layer_idx]['weights']

            # Confidence loss : encourage shar distributions per patch
            # We use entropy : Highet entropy = less confident, low entropy = more confident
            patch_entropies = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim = 1)
            confidence_loss = patch_entropies.mean()

            # Anti collapse loss
            experts_usage = (expert_weights > 0.01).float().mean(dim = 0)
            experts_unused = (experts_usage < 0.01).float().sum()
            collapse_loss = experts_unused 

            total_loss += (self.confidence_weight * confidence_loss) + \
                    (self.anticollapse * collapse_loss) 
            
            self.entropy_history.append(patch_entropies)
            self.confidence_loss_history.append(confidence_loss)
            self.collapse_loss_history.append(collapse_loss)
        
        total_loss = total_loss / len(router_cache) + 1
        
        return total_loss
    
    def calc_cache_metrics(self, router_cache_history):
        """
        Calculate statistics from cached router metrics.

        Args:
            metrics (dict): Dictionary of router metrics.

        Returns:
            dict: Dictionary of aggregated cache metrics.
        """
        batches_num = len(router_cache_history)
        num_layers = len(router_cache_history[0])

        stats = {}

        # Find key to extract
        keys = list(router_cache_history[0][0].keys())

        for layer_idx in range(num_layers):
            stats = [layer_idx] = {}

            for key in keys:
                # Extract all tensor of current metrics from all batches
                values = [router_cache_history[b_idx][layer_idx][key] for b_idx in range(batches_num)]

                # Concat all
                values_cat = torch.cat([v.detach().cpu() for v in values], dim = 0)
                stats[layer_idx][key] = {
                    'mean': values_cat.mean().item(),
                    'std': values_cat.std().item(),
                    'min': values_cat.min().item(),
                    'max': values_cat.max().item()
                }

        return stats
