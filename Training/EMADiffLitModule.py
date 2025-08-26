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

        # Validation losses
        self.val_class_losses = []
        self.val_router_losses = []
        self.val_total_losses = []

        # Best validation loss
        self.best_val_loss = float('+inf')

        self.aux_loss_weight = 0.002

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
        
        logits, aux_loss = self(data)
        aux_loss *= self.aux_loss_weight

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
        total_loss = class_loss + aux_loss

        self.train_class_losses.append(class_loss.item())
        self.train_aux_losses.append(aux_loss.item())
        self.train_total_losses.append(total_loss.item())
        
        if len(self.class_names) >= 5:
            self.accuracy_metrics['top1_train'].update(logits, labels)
            self.accuracy_metrics['top5_train'].update(logits, labels)

        return {'loss' : total_loss}

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

        # AUX_MIN, AUX_MID, AUX_MAX = 0.002, 0.0065, 0.010
        # # Set aux loss weight
        # if self.current_epoch <= 30:
        #     self.aux_loss_weight = 0.0
        # elif self.current_epoch < 40:
        #     t = (self.current_epoch - 30) / 10
        #     self.aux_loss_weight = AUX_MIN + t * (AUX_MID - AUX_MIN)
        # elif self.current_epoch < 79:
        #     t = (self.current_epoch - 40) / 30
        #     self.aux_loss_weight = AUX_MID + t * (AUX_MAX - AUX_MID)
        # else:
        #     self.aux_loss_weight = AUX_MAX

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
        }

        self.train_class_losses.clear()
        self.train_aux_losses.clear()
        self.train_total_losses.clear()

        router_metrics = self.model.moe_aggregator.finalize()
        log_dict.update(router_metrics)

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
        # data = self.augmentation(data)
        
        logits, aux_loss = self(data)
        aux_loss *= self.aux_loss_weight

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
        total_loss = class_loss + aux_loss

        self.val_class_losses.append(class_loss.item())
        self.val_aux_losses.append(aux_loss.item())
        self.val_total_losses.append(total_loss.item())

        if len(self.class_names) >= 5:
            self.accuracy_metrics['top1_val'].update(logits, labels)
            self.accuracy_metrics['top5_val'].update(logits, labels)

        return {'loss' : total_loss}

    def on_validation_epoch_start(self):
        self.accuracy_metrics['top1_val'].reset()
        self.accuracy_metrics['top5_val'].reset()

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

        router_metrics = self.model.moe_aggregator.finalize()
        log_dict.update(router_metrics)

        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

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