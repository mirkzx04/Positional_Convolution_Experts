import torch
import os

class Checkpointer:
    def __init__(self, check_point_dir):
        """
        Initialize the Checkpointer class.
        This class is used to save and load model checkpoints.
        """
        self.check_point_dir = check_point_dir

    def model_checkpoints(self, model):
    # Check if exist model checkpoint
        if os.path.exists(f'{self.check_point_dir}/model_checkpoints.pth'):
            print(f'Loading model from checkpoint...')
            model.load_state_dict(torch.load(f'{self.check_point_dir}/model_checkpoints.pth'))

            print('Model loaded from checkpoint')

        else:
            print('No model checkpoint founded')

        return model
    
    def train_checkpoints(self):
    # Check if exist train params checkpoint, included scheduler and optimizer
        if os.path.exists(f'{self.check_point_dir}/train_checkpoints.pth'):
            print('Load training params from checkpoint...')
            checkpoint = torch.load(f'{self.check_point_dir}/train_checkpoints.pth')

            start_epoch = checkpoint['start_epoch']
            start_train_batch = checkpoint['train_batch']

            train_loss_history = checkpoint['train_history']
            val_loss_history = checkpoint['val_history']

            optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = lr_scheduler.load_state_dict(checkpoint['scheduler'])

            print(f'Resuming train from {start_epoch} epoch and {start_train_batch} train batch')
            
            return start_epoch, start_train_batch, train_loss_history, val_loss_history, optimizer, lr_scheduler
        else:
            print('No train checkpoint founded')

        return 0, 0, [], [], None, None

    def save_backbone_checkpoint(self, model, optimizer, lr_scheduler, 
                   epoch, batch_idx, train_loss_history, val_loss_history):
        """Save model and training state"""

        if not os.path.exists(self.check_point_dir):
            os.makedirs(self.check_point_dir)
        
        # Save model
        torch.save(model.state_dict(), f'{self.check_point_dir}/model_checkpoints.pth')
        
        # Save training state
        torch.save({
            'start_epoch': epoch,
            'train_batch': batch_idx,
            'train_history': train_loss_history,
            'val_history': val_loss_history,
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict()
        }, f'{self.check_point_dir}/backbone_checkpoints.pth')
    
    def get_start_epoch(self):
        if os.path.exists(f'{self.check_point_dir}/train_checkpoints.pth'):
            print('Load training params from checkpoint...')
            checkpoint = torch.load(f'{self.check_point_dir}/train_checkpoints.pth')

            start_epoch = checkpoint['start_epoch']

            return start_epoch
        else:
            return 0