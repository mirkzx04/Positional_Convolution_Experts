import pytorch_lightning as pl
import torch

from torch.nn import functional as F
from torch.optim import Adam

from torchmetrics import Accuracy

from PCEScheduler import PCEScheduler

class EMADiffLitModule(pl.LightningModule):
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
                phase_multipliers,
                backbone_epochs,
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

        self.backbone_epochs = backbone_epochs

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_class_losses = []
        self.train_router_losses = []
        self.train_total_losses = []

        self.val_class_losses = []
        self.val_router_losses = []
        self.val_total_losses = []

        self.best_val_loss = '-inf'

        self.accuracy_metrics = {
            'top1_train' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1),
            'top5_train' : Accuracy(task='multiclass', num_classes=num_classes, top_k=5),

            'top1_val' : Accuracy(task='multiclass', num_classes=num_classes, top_k=1),
            'top5_val' : Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        }

        self.confidence_weight = 0.01, 
        self.anticollapse = 0.001,
        self.router_metrics = False

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        if self.current_epoch < self.start_epoch and batch_idx < self.start_train_batch:
            return None
        
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)

        logits = self(data)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        class_loss = self.criterion(logits, labels)

        # Compute router loss
        router_loss = self.router_loss()
        
        total_loss = class_loss.item() + router_loss.item()

        self.train_class_losses.append(class_loss.item())
        self.train_router_losses.append(router_loss.item())
        self.train_total_losses.append(total_loss.item())

        if self.num_classes >= 5:
            self.accuracy_metrics['top1_train'].update(logits, labels)
            self.accuracy_metrics['top5_train'].update(logits, labels)

        return {
            'pred_labels': predictions,
            'class_loss': class_loss,
            'router_loss': router_loss,
            'loss' : total_loss,
            'actual_batch' : batch_idx
        }
    
    def on_train_epoch_end(self, outputs, batch, batch_idx):
        self.avg_train_class_losses = torch.tensor(self.train_class_losses).mean().item()
        self.avg_train_router_losses = torch.tensor(self.train_router_losses).mean().item()
        self.avg_train_total_losses = torch.tensor(self.train_total_losses).mean().item()

        self.train_class_losses.clear()
        self.train_router_losses.clear()
        self.train_total_losses.clear()

    def validation_step(self, batch, batch_idx):        
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

        if self.num_classes >= 5:
            self.accuracy_metrics['top1_val'].update(logits, labels)
            self.accuracy_metrics['top5_val'].update(logits, labels)

        return {
            'pred_labels': predictions,
            'class_loss': class_loss,
            'router_loss': router_loss,
            'loss' : total_loss,
        }

    def on_validation_epoch_end(self):
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
            'avg_train_class_loss' : self.avg_train_class_losses,
            'avg_train_router_loss' : self.avg_train_router_losses,
            'avg_train_total_loss' : self.train_total_losses,
            'train_top1_acc' : self.accuracy_metrics['top1_train'].compute().item(),
            'train_top5_acc' : self.accuracy_metrics['top5_train'].compute().item(),
            'avg_val_classification_loss' : self.avg_val_class_losses,
            'avg_val_router_loss' : self.avg_val_router_losses,
            'avg_val_total_loss' : self.avg_val_total_losses,
            'val_top1_acc' : self.accuracy_metrics['top1_val'].compute().item(),
            'val_top5_acc' : self.accuracy_metrics['top5_val'].compute().item(),
            'epoch' : self.current_epoch,
            'gradient_norm' : self.calculate_gradient_norm(),
            'best_val_loss' : self.best_val_loss,
            'router_metrics' : routing_metrics
        }

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
            if param.grad is not None:
                param_norm = torch.norm(param.grad.data).item()
                total_grad_norm += param_norm ** 2
        
        global_grad_norm = total_grad_norm ** 0.5
        return global_grad_norm
    
    def router_loss(self):
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
        cache_metrics = {}
        all_cache_metrics = []

        for key, value in metrics.items():
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