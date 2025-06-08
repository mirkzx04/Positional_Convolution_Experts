import torch.nn as nn
import wandb
import torch
import os 

from torch.optim import Adam
from tqdm import tqdm

from PCEScheduler import PCEScheduler

from Model.Components.Router import Router

from Datasets_Classes.PatchExtractor import PatchExtractor

class TrainModel:
    def __init__(self,logger, train_config, model, device):
        """
        Initialize the TrainModel class.
        This class is responsible for training the model.

        Args:
            datasets (dic): dictionaru containing all datasets and their dataloaders.
            logger (wandb.Logger): Logger for tracking experiments.
            cofig (dict): Configuration dictionary containing training parameters.
        """
        self.device = device
        self.model = model
        self.logger = logger
        self.train_config = train_config

        # Settings loss function, optimizer and LR scheduler
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = Adam(
            model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        self.LR_scheduler = PCEScheduler(
            optimizer = self.optimizer,
            phase_epochs = [
                self.config['pre_train_epochs'],
                self.config['fine_tune_epochs']
            ],
            base_lr= self.config['lr'],
            last_epoch= -1,
        )

    def check_checkpoints(self, train_checkpoints_path):
        """
        Check if the checkpoint directory exists, if not create it.

        Args:
            train_checkpoints_path (str): Path to save and load model checkpoints.
        """

        # Check if the model checkpoints exists
        if os.path.exists(f'{train_checkpoints_path}/model_checkpoints.pth'):
            print("Loading model from checkpoint...")
            self.model.load_state_dict(torch.load(f'{train_checkpoints_path}/model.pth'))
            print("Model loaded successfully.")
        else:
            print("No checkpoint found, starting training from scratch...")

        # Check if the training checkpoints exists
        if os.path.exists(f'{train_checkpoints_path}/train_checkpoints.pth'):
            print("Loading trainint checkpoints from checkpoint...")
            train_checkpoint = torch.load(f'{train_checkpoints_path}/train_checkpoints.pth')
            start_epoch = train_checkpoint['current_epoch']
            start_batch_training = train_checkpoint['batch_training']

            # Load losses history
            train_loss_history = train_checkpoint['train_loss_history']
            val_loss_history = train_checkpoint['val_loss_history']

            # Load the optimizer and LR scheduler state
            self.optimizer.load_state_dict(train_checkpoint['optimizer'])
            self.LR_scheduler.load_state_dict(train_checkpoint['scheduler'])

            print(f'Resuming training from epoch {start_epoch}, '
                  f'training batch {start_batch_training}, '
                  f'validation batch {start_batch_validation}...')
            
        else:
            print("No training checkpoints found, starting training from scratch...")
            start_epoch = 0
            start_batch_training = 0
            start_batch_validation = 0

            train_loss_history = []
            val_loss_history = []

        return start_epoch, start_batch_training, train_loss_history, val_loss_history
    
    def train(self, training_loader, validation_loader, train_checkpoints_path = './checkpoints'):
        """
        Train the model using the provided datasets and configurations.

        Args:
            train_checkpoints_path (str): Path to save and load model checkpoints.
            training_loader (DataLoader): DataLoader for the training dataset.
            validation_loader (DataLoader): DataLoader for the validation dataset.
        """
        # Set the datasets used for training and validation
        training_datasets = training_loader
        validation_loader = validation_loader

        # Check if the checkpoint directory exists, if not create it.
        start_epoch, start_batch_train, train_loss_history, val_loss_history  = self.check_checkpoints(train_checkpoints_path)
        
        self.model.to(self.device)

        # Training loop
        for epoch in tqdm(range(start_epoch, self.config['epochs']), desc="Training Epochs"):
            # Set model to training mode and reset losses
            self.model.train()

            train_loss = 0
            val_loss = 0

            training_batches = list(enumerate(training_datasets))
            current_batch_start = start_batch_train if epoch == start_epoch else 0

            if current_batch_start > 0:
                training_batches = training_batches[current_batch_start:]
                print(f'Resuming training from batch {current_batch_start}...')
            
            # Training phase
            for batch_idx, batch in tqdm(training_batches, desc="Training Batches"):
                self.upload_checkpoint(
                    current_epoch=epoch,
                    batch_training=actual_batch_idx,
                    train_loss_history=train_loss_history,
                    val_loss_history=val_loss_history
                )
                # Extract data and labels from the batch
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)

                # Forward pass
                logits = self.model(data)

                # Compute loss
                loss = self.criterion(logits, labels)
                train_loss += loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.LR_scheduler.step()

                train_loss_history.append(loss.item())

                actual_batch_idx = batch_idx + start_batch_train

            avg_train_loss = train_loss / len(training_batches)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in tqdm(validation_loader, desc="Validation Batches"):
                    # Extract data and labels from the batch
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)

                    # Forward pass
                    logits = self.model(data)

                    # Compute loss
                    loss = self.criterion(logits, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(validation_loader)

            val_loss_history.append(avg_val_loss)
            train_loss_history.append(avg_train_loss)

    def upload_checkpoint(self,
                          current_epoch, batch_training,
                          train_loss_history, val_loss_history):
        """
        Upload the current model and training state to the checkpoint directory.
        This function is called at the end of each epoch and batch to save the current state.

        Args:
            current_epoch (int): The current epoch number.
            batch_training (int): The current training batch number.
            batch_validation (int): The current validation batch number.
            train_loss_history (list): List of training loss values.
            val_loss_history (list): List of validation loss values.
        """
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')

        torch.save(self.model.state_dict(), './checkpoints/model.pth')
        torch.save({
            'current_epoch': current_epoch,
            'batch_training': batch_training,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.LR_scheduler.state_dict(),
        }, './checkpoints/train_checkpoints.pth')

