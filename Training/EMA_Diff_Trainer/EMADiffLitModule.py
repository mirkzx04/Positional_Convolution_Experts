import pytorch_lightning as pl
import torch
import wandb as wb

from torch.nn import functional as F
from torch.optim import Adam
from torchvision.transforms import ToPILImage
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchmetrics import Accuracy

from PCEScheduler import PCEScheduler
from Model.PCE import PCENetwork

class EMADiffLitModule(pl.LightningModule):
    def __init__(self, 
                num_experts,
                layer_number,
                patch_size,
                dropout,
                num_classes,
                nucleus_sampling_p,
                lr, 
                weight_decay,
                augmentation,
                class_names,
                device,
                train_epochs,
                log_every_batch = 100
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

        self.model = PCENetwork(
            num_experts = num_experts,
            layer_number = layer_number,
            patch_size = patch_size,
            dropout=dropout,
            num_classes=num_classes,
            nucleus_sampling_p=nucleus_sampling_p
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.augmentation = augmentation
        self.class_names = class_names
        self.device = device
        self.log_every_batch = log_every_batch
        self.train_epochs = train_epochs

        # Training losses
        self.train_class_losses = []
        self.train_router_losses = []
        self.train_total_losses = []
        self.train_importance_losses = []
        self.train_load_losses = []
        self.train_confidence_losses = []
        self.train_experts_usage = []

        # Validation losses
        self.val_class_losses = []
        self.val_router_losses = []
        self.val_total_losses = []
        self.val_importance_losses = []
        self.val_load_losses = []
        self.val_confidence_losses = []
        self.val_experts_usage = []

        # Router cache history
        self.train_router_cache_history = []
        self.val_router_cache_history = []

        # Best validation loss
        self.best_val_loss = float('+inf')

        self.aux_loss_weight = 0.0

        # Accuracy metrics
        self.accuracy_metrics = {
            'top1_train' : Accuracy(task='multiclass', num_classes=len(class_names), top_k=1).to(device),
            'top5_train' : Accuracy(task='multiclass', num_classes=len(class_names), top_k=5).to(device),

            'top1_val' : Accuracy(task='multiclass', num_classes=len(class_names), top_k=1).to(device),
            'top5_val' : Accuracy(task='multiclass', num_classes=len(class_names), top_k=5).to(device)
        }
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output logits from the model.
        """
        return self.model(x, current_epoch=self.current_epoch)
    
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
        data = self.augmentation(data)
        
        logits = self(data)

        if self.log_every_batch % batch_idx == 0:
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            self.log_prediction_to_wandb(
                data, 
                labels, 
                predictions, 
                self.class_names, 
                num_images_to_log=10, 
                epoch=self.current_epoch, 
                phase='train'
            )

        class_loss = self.criterion(logits, labels)
        self.train_class_losses.append(class_loss.item())

        # Compute router loss
        if self.current_epoch >= 30:
            # Get router cache
            router_cache = self.model.router.get_cache()
            router_loss = self.router_loss(router_cache, True)
            self.model.router.clear_cache()

            # Compute total loss
            total_loss = class_loss + (self.aux_loss_weight * router_loss)
            self.train_router_losses.append(router_loss.item())
            self.train_total_losses.append(total_loss.item())

            # Detach cache
            detach_cache = self.detach_cache(router_cache)
            self.train_router_cache_history.append(detach_cache)

            loss = total_loss
        else:
            # Compute class loss
            loss = class_loss
        
        if len(self.class_names) >= 5:
            self.accuracy_metrics['top1_train'].update(logits, labels)
            self.accuracy_metrics['top5_train'].update(logits, labels)

        return {'loss' : loss}

    def on_after_backward(self):
        """
        Calculate the gradient norm after backward pass.

        Returns:
            None
        """
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        self.gradient_norm = total_norm ** 0.5
    
    def on_train_epoch_start(self):
        self.accuracy_metrics['top1_train'].reset()
        self.accuracy_metrics['top5_train'].reset()

        # Set router trainable
        if self.current_epoch == 0:
            self.disable_router()
        elif self.current_epoch >= 30:
            self.active_router()

        # Set aux loss weight
        if self.current_epoch < 30:
            self.aux_loss_weight = 0.0
        elif 30 <= self.current_epoch < 40:
            t = (self.current_epoch - 30) / 10
            self.aux_loss_weight = 0.02 + t * (0.05 - 0.02)
        elif 40 < self.current_epoch < 50:
            t = (self.current_epoch - 40) / 10
            self.aux_loss_weight = 0.05 + t * (0.1 - 0.05)
        elif 40 <= self.current_epoch < 70:
            t = (self.current_epoch - 40) / 30
            self.aux_loss_weight = 0.1 + t * (0.15 - 0.1)
        else:
            self.aux_loss_weight = 0.15

        # if self.current_epoch < 30:
        #     value = 0.1
        # elif 30 <= self.current_epoch <= 50:
        #     t = (self.current_epoch - 30) / (50 - 30)  # 0 → 1
        #     value = 0.1 + t * (0.3 - 0.1)
        # else:
        #     value = 0.3

        # # Update all thresholds of experts
        # for layer in self.model.layers:
        #     if hasattr(layer, "threshold"):
        #         layer.threshold.data.fill_(value)

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch to compute and reset average losses.

        Returns:
            None
        """

        # Log dictionary
        log_dict = {
            'train_class_loss' : torch.tensor(self.train_class_losses).mean().item(),
            'train_router_loss' : torch.tensor(self.train_router_losses).mean().item(),
            'train_top1' : self.accuracy_metrics['top1_train'].compute().item() * 100,
            'train_top5' : self.accuracy_metrics['top5_train'].compute().item() * 100,
        }

        # If router is enabled, log router loss and cache metrics
        if self.current_epoch >= 30:
            log_dict['train_router_loss'] = torch.tensor(self.train_router_losses).mean().item()
            log_dict['train_total_loss'] = torch.tensor(self.train_total_losses).mean().item()
            log_dict['train_cache_stats'] = self.calc_cache_metrics(self.train_router_cache_history)

            log_dict['train_importance_loss'] = torch.tensor(self.train_importance_losses).mean().item()
            log_dict['train_load_loss'] = torch.tensor(self.train_load_losses.detach().cpu()).mean().item()

            log_dict['train_confidence_loss'] = torch.tensor(self.train_confidence_losses.detach().cpu()).mean().item()
            log_dict['train_experts_usage'] = torch.tensor(self.train_experts_usage.detach().cpu()).mean().item()

            self.train_router_losses.clear(),
            self.train_total_losses.clear()
            self.train_router_cache_history.clear()

            self.train_importance_losses.clear()
            self.train_load_losses.clear()
    
            self.train_confidence_losses.clear()
            self.train_experts_usage.clear()

        self.train_class_losses.clear(),  

        self.log(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

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
        # data = self.augmentation(data)
        
        logits = self(data)

        if self.log_every_batch % batch_idx == 0:
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            self.log_prediction_to_wandb(
                data, 
                labels, 
                predictions, 
                self.class_names, 
                num_images_to_log=10, 
                epoch=self.current_epoch, 
                phase='val'
            )

        class_loss = self.criterion(logits, labels)
        self.val_class_losses.append(class_loss.item())

        # Compute router loss
        if self.current_epoch >= 30:
            # Get router cache
            router_cache = self.model.router.get_cache()
            router_loss = self.router_loss(router_cache, train=False)
            self.model.router.clear_cache()

            # Compute total loss
            total_loss = class_loss + (self.aux_loss_weight * router_loss)
            self.val_router_losses.append(router_loss.item())
            self.val_total_losses.append(total_loss.item())

            # Detach cache
            detach_cache = self.detach_cache(router_cache)
            self.val_router_cache_history.append(detach_cache)

            loss = total_loss
        else:
            # Compute class loss
            loss = class_loss

        if len(self.class_names) >= 5:
            self.accuracy_metrics['top1_val'].update(logits, labels)
            self.accuracy_metrics['top5_val'].update(logits, labels)

        return {'loss' : loss}

    def on_validation_epoch_start(self):
        self.accuracy_metrics['top1_val'].reset()
        self.accuracy_metrics['top5_val'].reset()

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to compute and reset average losses.

        Returns:
            None
        """
        # Log dictionary
        log_dict = {
            'val_class_loss' : torch.tensor(self.val_class_losses).mean().item(),
            'val_top1' : self.accuracy_metrics['top1_val'].compute().item() * 100,
            'val_top5' : self.accuracy_metrics['top5_val'].compute().item() * 100,
        }

        # If router is enabled, log router loss and cache metrics
        if self.current_epoch >= 30:
            log_dict['val_router_loss'] = torch.tensor(self.val_router_losses).mean().item()
            log_dict['val_total_loss'] = torch.tensor(self.val_total_losses).mean().item()
            log_dict['val_cache_stats'] = self.calc_cache_metrics(self.val_router_cache_history)

            log_dict['val_importance_loss'] = torch.tensor(self.val_importance_losses).mean().item()
            log_dict['val_load_loss'] = torch.tensor(self.val_load_losses).mean().item()

            log_dict['val_confidence_loss'] = torch.tensor(self.val_confidence_losses).mean().item()
            log_dict['val_experts_usage'] = torch.tensor(self.val_experts_usage).mean().item()

            self.val_router_losses.clear()
            self.val_total_losses.clear()
            self.val_router_cache_history.clear()

            self.val_importance_losses.clear()
            self.val_load_losses.clear()

            self.val_confidence_losses.clear()
            self.val_experts_usage.clear()

        self.val_class_losses.clear()

        self.log(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # Router parameters
        router_params = list(self.model.router.parameters())

        # Backbone parameters = layers + threshold + conv finali ecc.
        router_ids = {id(p) for p in router_params}
        backbone_params = [p for p in self.model.parameters() if id(p) not in router_ids]

        base_lr = self.lr
        router_mul = 5.0
        wd = self.weight_decay

        # Optimizer
        self.optimizer = Adam(
            [
                {'params': backbone_params, 'lr': base_lr, 'weight_decay': wd, 'name' : 'backbone'},
                {'params': router_params, 'lr': base_lr * router_mul, 'weight_decay': wd, 'name' : 'router'}
            ]
        )

        # Learning rate scheduler
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.train_epochs, eta_min=0.00001)

        return [self.optimizer], [self.lr_scheduler]
    
    def router_loss(self, router_cache, train=False):
        """
        Compute the router loss, including confidence and anti-collapse losses.

        Args:
            router_cache: Cache from router containing expert weights and other metrics
            train (bool): Whether this is training or validation phase

        Returns:
            Tensor or tuple: Router loss tensor, and optionally router metrics and cache metrics if self.router_metrics is True.
        """

        total_losses_step = 0.0
        importance_losses_step= []
        load_losses_step = []

        confidence_losses_step = []
        experts_usage_list_step = []

        importance_loss = 0.0
        top1 = 0.0
        load_loss = 0.0

        patch_entropies = 0.0
        confidence_loss = 0.0
        experts_usage = 0.0

        total_loss = 0.0
        
        if not router_cache:
            return torch.tensor(0.0, requires_grad=True)
        
        for layer_idx in range(len(router_cache)):
            # Get experts weights [B x P, num_experts]
            expert_weights = router_cache[layer_idx]['norm_weights']

            # Importance loss
            importance = expert_weights.sum(dim = 0)
            importance_loss = ((importance.std() / importance.mean()) ** 2) / self.num_experts
            importance_losses_step.append(importance_loss.item())

            # Load loss
            top1 = expert_weights.argmax(dim = 1)
            load = torch.bincount(top1, minlength = expert_weights.shape[1]).float()
            load_loss = ((load.std() / load.mean()) ** 2) / self.num_experts
            load_losses_step.append(load_loss.item())

            with torch.no_grad():
                # Compute confidence loss
                expert_weights = expert_weights.detach().cpu()
                
                patch_entropies = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim = 1)
                confidence_loss = patch_entropies.mean().item()
                confidence_losses_step.append(confidence_loss)

                # Compute anti-collapse loss
                experts_usage = (expert_weights > 0.01).float().mean(dim = 0).item()    
                experts_usage_list_step.append(experts_usage)
            
            # Total loss
            total_loss += (0.5 * importance_loss) + (0.5 * load_loss)

        # Compute mean loss
        total_loss = total_loss / len(router_cache)

        with torch.no_grad():
            if train:
                # Compute mean importance loss
                self.train_importance_losses.append(torch.tensor(importance_losses_step).mean().item())
                # Compute mean load loss
                self.train_load_losses.append(torch.tensor(load_losses_step).mean().item())
                # Compute mean confidence loss
                self.train_confidence_losses.append(torch.tensor(confidence_losses_step).mean().item())
                # Compute mean experts usage
                self.train_experts_usage.append(torch.tensor(experts_usage_list_step).mean().item())
            else:
                # Compute mean importance loss
                self.val_importance_losses.append(torch.tensor(importance_losses_step).mean().item())
                # Compute mean load loss
                self.val_load_losses.append(torch.tensor(load_losses_step).mean().item())
                # Compute mean confidence loss
                self.val_confidence_losses.append(torch.tensor(confidence_losses_step).mean().item())
                # Compute mean experts usage
                self.val_experts_usage.append(torch.tensor(experts_usage_list_step).mean().item())
        
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
            stats[layer_idx] = {}

            for key in keys:
                # Extract all tensor of current metrics from all batches
                values = [router_cache_history[b_idx][layer_idx][key] for b_idx in range(batches_num)]

                processed_valued = []
                for v in values : 
                    v_detached = v.detach().cpu()
                    if v_detached.dim() == 0:
                        v_detached = v_detached.unsqueeze(0)
                    processed_valued.append(v_detached)
                
                values_cat = torch.cat(processed_valued, dim = 0)
                stats[layer_idx][key] = {
                    'mean': values_cat.mean().item(),
                    'std': values_cat.std().item(),
                    'min': values_cat.min().item(),
                    'max': values_cat.max().item()
                }

        return stats

    def log_prediction_to_wandb(self, 
        data_batch, 
        true_labels,
        pred_labels, 
        class_names, 
        num_images_to_log = 10, 
        epoch = None, 
        phase = 'train'):
            """
            Log model predictions to wandb with images, true labels and prediction labels

            Args:
                data_batch (torch.Tensor): Batch of images with shape (B, C, H, W)
                true_labels  // : True labels with shape (B,)
                pred_labels // : Predicted labels with shape (B,) or (B, num_classes)
                class_names ( list) : List of class name for CIFAR-10/other datasets
                num_images (int) : Number of images to log
                epoch // : Current epoch number
                step // : Current step number
                phase (str) : Training phase ('Training' or 'Validation')
                batch_loss (float) : loss of the current batch
            """

        # Convert tensors to numpy if needed
        if isinstance(data_batch, torch.Tensor):
            data_batch = data_batch.detach().cpu()
        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.detach().cpu()
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.detach().cpu()

        if len(pred_labels.shape) > 1:
            pred_labels = torch.argmax(pred_labels, dim = 1)
        
        # Limit number of images to log
        num_images_to_log = min(num_images_to_log, data_batch.shape[0])
        wandb_images = []

        for i in range(num_images_to_log):
            # Get singles image and labels
            img = data_batch[i]
            true_label = true_labels[i]
            pred_label = pred_labels[i]

            # Conver images tensor to PIL Image, handle different image formats (CHW vs HWC)
            if img.shape[0] == 3: 
                img_pil = ToPILImage()(img)
            else: #HWC format, transpose to CHW
                img = img.permute(2, 0, 1)
                img_pil = ToPILImage()(img)

            # Create caption with true and predicted labels
            true_class = class_names[true_label] if true_label < len(class_names) else f'True classes : {true_label}'
            pred_class = class_names[pred_label] if pred_label < len(class_names) else f'Predicted classes : {pred_label}'

            is_correct = "✓" if true_label == pred_label else "✗"
            caption = f'{is_correct} True : {true_class} | Pred : {pred_class} \n'

            # Create wandb image object
            wandb_img = wb.Image(
                img_pil,
                caption=caption
            )

            wandb_images.append(wandb_img)
        
        # Log to wandb
        log_dict = {f'{phase}_predictions' : wandb_images}
        self.log(log_dict, step = epoch)

    def detach_cache(self, router_cache):
        detached_router_cache = []
        for layer_cache in router_cache:
            detached_cache = {
                key : value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in layer_cache.items()
            }
            detached_router_cache.append(detached_cache)
        
        return detached_router_cache

    def active_router(self):
        for p in self.model.router.parameters():
            p.requires_grad_(True)
        
        for gate in self.model.router.getes:
            for p in gate.parameters():
                p.requires_grad_(True)
    
    def disable_router(self):
        for p in self.model.router.parameters():
            p.requires_grad_(False)
        
        for gate in self.model.router.getes:
            for p in gate.parameters():
                p.requires_grad_(False)