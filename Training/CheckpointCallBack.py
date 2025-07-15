import pytorch_lightning as pl

class CheckpointCallBack(pl.Callback):
    def __init__(self, checkpointer, save_checkpoint_every=5):
        super().__init__()
        self.checkpointer = checkpointer
        self.save_checkpoint_every = save_checkpoint_every
    
    def on_train_batch_end(self, trainer, pl_module, batch_idx):
        """
        Save model checkpoints at the end of each training batch.
        """
        if (batch_idx + 1) % self.save_checkpoint_every == 0:
            # Recovering optimizer and lr_scheduler
            optimizer = trainer.optimizers[0] 
            lr_scheduler = trainer.lr_scheduler_configs[0]['scheduler']

            # Recovering history 
            train_loss_history = getattr(pl_module, 'train_loss_history')
            train_val_history = getattr(pl_module, 'train_val_history')
            actual_phase = getattr(pl_module, 'actual_phase')

            self.checkpointer.save_checkpoints(
                pl_module.model, optimizer, lr_scheduler, trainer.current_epoch, batch_idx, train_loss_history, train_val_history, actual_phase
            )
    
    def on_train_epoch_end(self, trainer, pl_module):

        # Recovering optimizer and lr_scheduler
        optimizer = trainer.optimizers[0] 
        lr_scheduler = trainer.lr_scheduler_configs[0]['scheduler']

        # Recovering history 
        train_loss_history = getattr(pl_module, 'train_loss_history')
        train_val_history = getattr(pl_module, 'train_val_history')
        actual_phase = getattr(pl_module, 'actual_phase')


        self.checkpointer.save_checkpoints(
            pl_module.model, optimizer, lr_scheduler, trainer.current_epoch, 0, train_loss_history, train_val_history, actual_phase 
        )


        
    