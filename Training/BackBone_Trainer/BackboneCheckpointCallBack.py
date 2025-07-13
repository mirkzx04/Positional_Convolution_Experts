import pytorch_lightning as pl

class BackboneCheckpointCallBack(pl.Callback):
    def __init__(self, checkpointer, save_checkpoint_every=5):
        super().__init__()
        self.checkpointer = checkpointer
        self.save_checkpoint_every = save_checkpoint_every
    
    def on_train_batch_end(self,trainer, model, optimizer, lr_scheduler, 
                   epoch, batch_idx, train_loss_history, val_loss_history):
        """
        Save model checkpoints at the end of each training batch.
        """
        if (batch_idx + 1) % self.save_checkpoint_every == 0:
            self.checkpointer.save_backbone_checkpoint(
                model, optimizer, lr_scheduler, 
                epoch, batch_idx, train_loss_history, val_loss_history
            )
        
    