import pytorch_lightning as pl

class EMADiffLoggerCallBack(pl.Callback):
    def __init__(self, logger, str_epoch, log_predicttion_every_batch=1):
        super().__init__()
        self.logger = logger
        self.log_predicttion_every_batch = log_predicttion_every_batch
        self.str_epoch = str_epoch

    def on_train_batch_end(self, trainer, pl_module, output, batch, batch_idx):
        """
        Log model predictions to wandb with images, true labels and prediction labels
        """
        if output:
            if (batch_idx + 1) % self.log_predicttion_every_batch == 0:
                data_batch, true_labels = batch
                
                pred_labels = output['pred_labels'] if isinstance(output, dict) and 'pred_labels' in output else None

                self.logger.log_prediction_to_wandb(
                    data_batch, 
                    true_labels, 
                    pred_labels, 
                    class_names=pl_module.class_names if hasattr(pl_module, 'class_names') else None,
                    num_images_to_log=10,
                    epoch=trainer.current_epoch,
                    phase='train_backbone'
                )

    def on_validation_batch_end(self, trainer, pl_module, output, batch, batch_idx):
        """
        Log model predictions to wandb with images, true labels and prediction labels
        """
        if output:
            if (batch_idx + 1) % self.log_predicttion_every_batch == 0:
                data_batch, true_labels = batch
                
                pred_labels = output['pred_labels'] if isinstance(output, dict) and 'pred_labels' in output else None

                self.logger.log_backbone_metrics_to_wandb(
                    data_batch, 
                    true_labels, 
                    pred_labels, 
                    class_names=pl_module.class_names if hasattr(pl_module, 'class_names') else None,
                    num_images_to_log=10,
                    epoch=trainer.current_epoch,
                    phase='train_backbone'
                ) 
    
    def on_validation_epoch_end(self, trainer, pl_module, output):
        if self.str_epoch > trainer.current_epoch:
            return None

        avg_train_class_loss = getattr(pl_module, 'avg_train_class_loss', None)
        avg_train_router_loss = getattr(pl_module, 'avg_train_router_loss', None)
        avg_train_total_loss = getattr(pl_module, 'avg_train_total_loss', None)
        train_top1_acc = getattr(pl_module, 'train_top1_acc', None)
        train_top5_acc = getattr(pl_module, 'train_top5_acc', None)
        avg_val_classification_loss = getattr(pl_module, 'avg_val_class_losses', None)
        avg_val_router_loss = getattr(pl_module, 'avg_val_router_losses', None)
        avg_val_total_loss = getattr(pl_module, 'avg_val_total_losses', None)
        val_top1_acc = getattr(pl_module, 'val_top1_acc', None)
        val_top5_acc = getattr(pl_module, 'val_top5_acc', None)
        lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else 0
        epoch = trainer.current_epoch
        gradient_norm = getattr(pl_module, "gradient_norm", None)
        best_val_loss = getattr(pl_module, "best_val_loss", None)
        router_metrics = getattr(pl_module, 'routing_log')

        self.logger.log_train_metrics_to_wandb(
            avg_train_class_loss, avg_train_router_loss, avg_train_total_loss, train_top1_acc,
            train_top5_acc, avg_val_classification_loss, avg_val_router_loss, 
            avg_val_total_loss, val_top1_acc, val_top5_acc, lr, epoch, gradient_norm, 
            best_val_loss, router_metrics
        )