import pytorch_lightning as pl

class BackboneLoggerCallBack(pl.Callback):
    def __init__(self, logger, log_predicttion_every_batch=1):
        super().__init__()
        self.logger = logger
        self.log_predicttion_every_batch = log_predicttion_every_batch

    def on_train_batch_end(self, trainer, pl_module, output, batch, batch_idx):
        """
        Log model predictions to wandb with images, true labels and prediction labels
        """
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
        if (batch_idx + 1) % self.log_predicttion_every == 0:
            data_batch, true_labels = batch
            
            pred_labels = output['pred_labels'] if isinstance(output, dict) and 'pred_labels' in output else None

            self.logger.log_prediction_to_wandb(
                data_batch, 
                true_labels, 
                pred_labels, 
                class_names=pl_module.class_names,
                num_images_to_log=10,
                epoch=trainer.current_epoch,
                phase='train_backbone'
            ) 
            
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Log training metrics to wandb at the end of each epoch.
        """

        avg_train_loss = getattr(pl_module, "avg_train_loss", None)
        avg_val_loss = getattr(pl_module, "avg_val_loss", None)
        train_top1_acc = getattr(pl_module, "train_top1_acc", None)
        train_top5_acc = getattr(pl_module, "train_top5_acc", None)
        val_top1_acc = getattr(pl_module, "val_top1_acc", None)
        val_top5_acc = getattr(pl_module, "val_top5_acc", None)
        lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else 0
        epoch = trainer.current_epoch
        gradient_norm = getattr(pl_module, "gradient_norm", None)
        best_val_loss = getattr(pl_module, "best_val_loss", None)

        self.logger.log_backbone_metrics_to_wandb(
            avg_train_loss, avg_val_loss,
            train_top1_acc, train_top5_acc,
            val_top1_acc, val_top5_acc,
            lr, epoch, gradient_norm, best_val_loss
        )
    