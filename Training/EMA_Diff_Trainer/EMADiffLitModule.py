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
        self.current_phase = actual_phase

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_class_losses = []
        self.train_router_losses = []
        self.train_total_losses = []

        self.val_class_losses = []
        self.val_router_losses = []
        self.val_total_losses = []

        self.best_val_loss = float('+inf')

        self.accuracy_metrics = {
            'top1_train' : Accuracy(task='multiclass', num_classes=len(class_names), top_k=1).to(device),
            'top5_train' : Accuracy(task='multiclass', num_classes=len(class_names), top_k=5).to(device),

            'top1_val' : Accuracy(task='multiclass', num_classes=len(class_names), top_k=1).to(device),
            'top5_val' : Accuracy(task='multiclass', num_classes=len(class_names), top_k=5).to(device)
        }

        self.confidence_weight = 0.01
        self.anticollapse = 0.001
        self.router_metrics = False

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
        router_loss_out = self.router_loss()
        if isinstance(router_loss_out, tuple):
            router_loss = router_loss_out[0]
        else:
            router_loss = router_loss_out

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
        self.gradient_norm = self.calculate_gradient_norm(self.model)
    
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
        self.avg_train_class_losses = torch.tensor(self.train_class_losses).mean().item()
        self.avg_train_router_losses = torch.tensor(self.train_router_losses).mean().item()
        self.avg_train_total_losses = torch.tensor(self.train_total_losses).mean().item()

        self.train_class_losses.clear()
        self.train_router_losses.clear()
        self.train_total_losses.clear()

        return {
            'train_loss_history' : self.train_loss_history,
            'val_loss_history' : self.val_loss_history,
            'actual_phase' : self.actual_phase
        }

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
        self.router_metrics = True
        router_loss, self.router_metrics, self.cache_metrics = self.router_loss()
        
        total_loss = class_loss.item() + router_loss.item()

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

    def on_validation_epoch_end(self):
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

        self.val_class_losses.clear()
        self.val_router_losses.clear()
        self.val_total_losses.clear()

        routing_metrics = self.combine_router_metrics(self.router_metrics, self.cache_metrics)

        if self.avg_val_total_losses < self.best_val_loss:
            self.best_val_loss = self.avg_val_total_losses

        return {
            'router_metrics' : routing_metrics
        }

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
    
    def router_loss(self):
        """
        Compute the router loss, including confidence and anti-collapse losses.

        Returns:
            Tensor or tuple: Router loss tensor, and optionally router metrics and cache metrics if self.router_metrics is True.
        """

        metrics = self.model.router.get_cahed_metrics()
        if metrics is None:
            return torch.tensor(0.0, requires_grad=True)
                
        # Get experts weights [B x P, num_experts]
        expert_weights = metrics['weights/weights_raw']

        # Confidence loss : encourage shar distributions per patch
        # We use entropy : Highet entropy = less confident, low entropy = more confident
        patch_entropies = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim = 1)
        confidence_loss = patch_entropies.mean()

        # Anti collapse loss
        experts_usage = (expert_weights > 0).float().mean(dim = 0)
        experts_unused = (experts_usage < 0.01).float().sum()
        collapse_loss = experts_unused 

        # Penalize conservative or permissive threshold
        # adaptive_threshold = metrics['adaptive_threshold']
        # threshold_extreme_penalty = torch.relu(adaptive_threshold - 0.7).mean() + \
        #                             torch.relu(0.05 - adaptive_threshold).mean()

        total_loss = total_loss = (self.confidence_weight * confidence_loss) + \
                (self.anticollapse * collapse_loss) 
        
        if self.router_metrics:
            router_metrics = {
                'router_loss/confidence_loss': confidence_loss.item(),
                'router_loss/collapse_loss': collapse_loss.item(),
                'router_loss/total_specialization_loss': total_loss.item(),
                'router_loss/avg_patch_entropy': patch_entropies.mean().item(),
                'router_loss/unused_experts_count': experts_unused.item(),
                'router_loss/min_expert_usage': experts_usage.min().item(),
                'router_loss/max_expert_usage': experts_usage.max().item(),

                # 'router_loss/threshold_extreme_penalty': threshold_extreme_penalty.item(),
                # 'router_loss/avg_adaptive_threshold': adaptive_threshold.mean().item(),
                # 'router_loss/threshold_std': adaptive_threshold.std().item(),
                # 'router_loss/threshold_range': (adaptive_threshold.max() - adaptive_threshold.min()).item(),
            }

            cache_metrics = self.calc_cache_metrics(metrics)

            return total_loss, router_metrics, cache_metrics
    
        return total_loss
    
    def calc_cache_metrics(self, metrics):
        """
        Calculate statistics from cached router metrics.

        Args:
            metrics (dict): Dictionary of router metrics.

        Returns:
            dict: Dictionary of aggregated cache metrics.
        """
        cache_metrics = {}
        all_cache_metrics = []

        for key, value in metrics.items():
            value = value.detach()
            if isinstance(value,torch.Tensor):
                if value.dim() == 1:
                    cache_metrics[key] = value.mean().item()
                else:
                    cache_metrics[f'cache_{key}_mean'] = value.mean().item()
                    cache_metrics[f'cache_{key}_std'] = value.std().item()  
                    cache_metrics[f'cache_{key}_max'] = value.max().item()
                    cache_metrics[f'cache_{key}_min'] = value.min().item()

                    if key == 'cosine_similarity':
                        cache_metrics[f'cache_{key}_range'] = (value.max() - value.min()).item()

                        sorted_sims = value.sort(dim=-1, descending=True)[0]
                        if sorted_sims.shape[-1] > 1:
                            top_gatp = (sorted_sims[:, 0] - sorted_sims[:, 1:]).mean().item()
                            cache_metrics[f'cache_top_gap'] = top_gatp
            else : 
                cache_metrics[f'cache_{key}'] = value
        all_cache_metrics.append(cache_metrics)

    def combine_router_metrics(self, cache_metrics, router_metrics):
        """
        Combine cache and router metrics into a single dictionary.

        Args:
            cache_metrics (list): List of cache metrics dictionaries.
            router_metrics (list): List of router metrics dictionaries.

        Returns:
            dict: Combined metrics dictionary.
        """
        combined_metrics = {}

        if cache_metrics:
            for key in cache_metrics[0].keys():
                combined_metrics[key] = sum(m[key] for m in cache_metrics) / len(cache_metrics)

        if router_metrics:
            all_keys = set()
            for m in cache_metrics:
                all_keys.update(m.keys())
            for key in all_keys:
                values = [m[key] for m in router_metrics if key in m]
                if values:
                    combined_metrics[key] = sum(values) / len(values)
        
        self.router_metrics = False
        return combined_metrics