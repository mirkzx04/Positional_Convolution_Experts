import wandb as wb

from torchvision.transforms import ToPILImage
import torch

class Logger:

    def __init__(self):
        """
        Initialize the Logger class.
        This class is used to log metrics and predictions to Weights & Biases (wandb).
        """
        self.logger = None
    
    def setup_wandb(self, 
                current_dataset,
                project_name="PCE",
                num_exp=4,
                kernel_size=3,
                out_channel_exp=8,
                out_channel_rout=20,
                layer_number=4,
                patch_size=16,
                lr=0.001,
                batch_size=32,
                epochs=10,
                dropout=0.1,
                ema_alpha=0.99,
                weight_decay=1e-4,
                threshold=0.5,
                router_temperature = 0.5
            ):
        
        wb.init(
            project="PCE",
            entity="mirkzx-sapienza-universit-di-roma",  # Replace with your WandB entity name
            config={
                'num_experts': num_exp,
                'kernel_size': kernel_size,
                'out_channel_exp': out_channel_exp,
                'out_channel_rout': out_channel_rout,
                'layer_number': layer_number,
                'patch_size': patch_size,
                'learning_rate': lr,
                'batch_size': batch_size,
                'epochs': epochs,
                'dropout': dropout,
                'ema_alpha': ema_alpha,
                'weight_decay': weight_decay,
                'threshold': threshold,
                'current dataset' : current_dataset,
                'router_temperature' : router_temperature
            }
        ) 

        self.logger = wb.config
    
    def log_prediction_to_wandb(self, data_batch, 
                            true_labels,
                            pred_labels, 
                            class_names = None, 
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
        wb.log(log_dict, step = epoch)
    
    def log_backbone_metrics_to_wandb(self, avg_train_loss, avg_val_loss,
                                      train_top1_acc, train_top5_acc,
                                      val_top1_acc, val_top5_acc,
                                      lr, epoch, gradient_norm,
                                      best_val_loss):
        """
        Log backbone training metrics to wandb

        Args:
            avg_train_loss (float): Average training loss
            avg_val_loss (float): Average validation loss
            train_top1_acc (float): Training top-1 accuracy
            train_top5_acc (float): Training top-5 accuracy
            val_top1_acc (float): Validation top-1 accuracy
            val_top5_acc (float): Validation top-5 accuracy
            lr (float): Current learning rate
            epoch (int): Current training epoch
            gradient_norm (float): Norm of gradient
            best_val_loss (float): Best validation loss
        """
        log_dict = {
            'back_bone/avg_train_loss': avg_train_loss,
            'back_bone/avg_val_loss': avg_val_loss,
            'back_bone/train_top1_acc': train_top1_acc,
            'back_bone/train_top5_acc': train_top5_acc,
            'back_bone/val_top1_acc': val_top1_acc,
            'back_bone/val_top5_acc': val_top5_acc,   
            'back_bone/lr': lr,
            'back_bone/epoch': epoch,
            'back_bone/gradient_norm': gradient_norm,
            'back_bone/best_val_loss': best_val_loss,
        }

        wb.log(log_dict, step=epoch)

    def log_train_metrics_to_wandb(
        self, avg_train_class_loss, avg_train_router_loss, avg_train_total_loss, train_top1_acc,
        train_top5_acc, avg_val_classification_loss, avg_val_router_loss, 
        avg_val_total_loss, val_top1_acc, val_top5_acc, lr, epoch, gradient_norm, 
        best_val_loss, router_metrics
        ):
        """
        Log train metrics to wandb
        
        avg_train_class_loss=avg_train_class_loss,
                avg_train_router_loss=avg_train_router_loss,
                avg_train_total_loss=avg_train_total_loss,
                train_top1_acc=train_top1_acc,
                train_top5_acc = train_top5_acc,
                avg_val_classification_loss=avg_val_classification_loss,
                avg_val_router_loss=avg_val_router_loss,
                avg_val_total_loss=avg_val_total_loss,
                val_top1_acc=val_top1_acc,
                val_top5_acc = val_top5_acc,
                lr=lr,
                epoch=epoch,
                gradient_norm=gradient_norm,
                best_val_loss=best_val_loss,
                router_metrics = router_metrics

        Args:
        avg_train_class_loss (float): Average training classification loss
        avg_train_router_loss (float): Average training router loss
        avg_train_total_loss (float): Average training total loss
        train_accuracy (float): Training accuracy
        avg_val_classification_loss (float): Average validation classification loss
        avg_val_router_loss (float): Average validation router loss
        avg_val_total_loss (float): Average validation total loss
        val_accuracy (float): Validation accuracy
        lr (float): Current learning rate
        epoch (int): Current training epoch
        gradient_norm (float): Norm of gradient
        best_val_loss (float): Best validation loss
        """

        log_dict = {
            'avg_train_class_loss': avg_train_class_loss,
            'avg_train_router_loss': avg_train_router_loss,
            'avg_train_total_loss': avg_train_total_loss,
            'train_top1_acc': train_top1_acc,
            'train_top5_acc' : train_top5_acc,
            'avg_val_classification_loss': avg_val_classification_loss,
            'avg_val_router_loss': avg_val_router_loss,
            'avg_val_total_loss': avg_val_total_loss,
            'val_top1_acc': val_top1_acc,
            'val_top5_acc' : val_top5_acc,
            'lr': lr,
            'epoch': epoch,
            'gradient_norm': gradient_norm,
            'best_val_loss': best_val_loss,
        }

        # Adding router metrics if avaible
        if router_metrics:
            log_dict.update(router_metrics)
        
        wb.log(log_dict, step=epoch)