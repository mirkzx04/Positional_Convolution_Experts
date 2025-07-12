import torch
import tqdm

from functools import partial
from contextlib import nullcontext

from torch.amp import autocast, GradScaler

from Checkpointer import Checkpointer
from Logger import setup_wandb, log_train_metrics_to_wandb, log_prediction_to_wandb


class Trainer:
    def __init__(self, model, optimizer, scheduler, device, save_checkpoint_every=5, log_predicttion_every = 100):
        self.checkpointer = Checkpointer(check_point_dir='../checkpoints')

        self.device = device
        self.model = self.checkpointer.model_checkpoints(model).to(device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.save_checkpoint_every = save_checkpoint_every  # Save checkpoint every 5 epochs

    def train_backbone(self, train_loader, val_loader, epochs):
        """
        Train the backbone of the model.
        This method trains the model for a specified number of epochs using the provided data loaders.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            epochs (int): Number of epochs to train the model.
        """
        start_epoch, start_train_batch, train_loss_history, val_loss_history, self.optimizer, self.lr_scheduler = self.check_backbone_checkpoints()
        use_amp, scaler = self.mix_precision_context()

        criterion = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(start_epoch, epochs), desc='Training Backbone'):
            self.model.train()
            train_loss = 0.0

            for batch_idx, (data, target) in enumerate(tqdm(train_loader, dec= 'Training Batch')):
                if epoch < start_epoch and batch_idx < start_train_batch:
                    continue

                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss =criterion(output, target)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss / len(train_loader.dataset)
            train_loss_history.append(train_loss)

            # Validation
            val_loss = self.validate(val_loader)

            val_loss_history.append(val_loss)

            # Log metrics
            log_train_metrics_to_wandb(epoch, train_loss, val_loss, self.lr_scheduler.get_last_lr()[0])

            # Save checkpoints
            if batch_idx % self.save_checkpoint_every == 0:
                self.checkpointer.save_backbone_checkpoints(
                    epoch + 1, batch_idx + 1, train_loss_history, val_loss_history, self.optimizer, self.lr_scheduler
                )

            # Log predictions to wandb
            if (epoch + 1) % self.save_checkpoint_every == 0:
                log_prediction_to_wandb(self.model, val_loader, epoch + 1, self.device, log_every=self.log_predicttion_every)

    def check_backbone_checkpoints(self):
        start_epoch, start_train_batch, train_loss_history, \
        val_loss_history, optimizer, lr_scheduler = self.checkpointer.backbone_checkpoints(self.optimizer, self.scheduler)
        if start_epoch != 0:
            print(f'Resume training from epoch number : {start_epoch}')
        if start_train_batch != 0:
            print(f'Resume batches from batch_idx : {start_train_batch}')
        print('\n ------------------------ \n')

        return start_epoch, start_train_batch, train_loss_history, val_loss_history, optimizer, lr_scheduler
    
    def mix_precision_context(self):
        """
        Context manager for mixed precision training.
        Returns a context manager that enables mixed precision training if available.
        """
        use_amp = torch.cuda.is_available() and str(self.device).startswith('cuda')
        scaler = GradScaler("cuda") if use_amp else None
        autocast_ctx = partial(autocast, "cuda") if use_amp else nullcontext
        print(f"Mixed precision enabled: {use_amp}")
        print('\n ------------------------ \n')

        return use_amp, scaler
    
    def calculate_gradient_norm(model):
        """
        Calculate norm of gradient
        
        Args:
            model: pytorch model
            
        Returns:
            float: Global norm of gradient
        """
        total_grad_norm = 0.0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = torch.norm(param.grad.data).item()
                total_grad_norm += param_norm ** 2
        
        global_grad_norm = total_grad_norm ** 0.5
        return global_grad_norm