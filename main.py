import torch
import wandb as wb
from tqdm import tqdm

import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from Datasets_Classes.Cifar10 import CIFAR10Dataset, CIFAR10TrainDataset, CIFAR10ValidationDataset
from Datasets_Classes.TinyImageNet import TinyImageNetDataset, TinyImageNetTrainDataset, TinyImageNetValidationDataset
# from Datasets_Classes.PascalVOC import PascalVOCDataset, PascalVOCTrainDataset, PascalVOCValidationDataset
from Datasets_Classes.PatchExtractor import PatchExtractor

from Model.PCE import PCENetwork
from Model.Components.Router import Router

from PCEScheduler import PCEScheduler

from DataAugmentation import DataAgumentation

def download_tiny_imagenet():
    """
    Download the Tiny ImageNet dataset.
    """
    import os
    import requests
    from zipfile import ZipFile

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    tiny_dir = "./Data/tiny-imagenet-200"

    if os.path.exists(tiny_dir):
        print("Tiny ImageNet dataset already exists.")
        return tiny_dir
    
    print("Downloading Tiny ImageNet dataset...")

    try:
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")

        print("Extracting Tiny ImageNet dataset...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path="./Data")

        os.remove(filename)  # Remove the zip file after extraction
        print("Extraction complete.")
        return tiny_dir
    except Exception as e:
        print(f"An error occurred while downloading or extracting the dataset: {e}")
        return None

def download_pascal_voc():
    """
    Download the Pascal VOC dataset.
    """
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    filename = "VOCtrainval_11-May-2012.tar"
    pascal_dir = "./Data/Pascal_VOC"

    if os.path.exists(pascal_dir):
        print("Pascal VOC dataset already exists.")
        return pascal_dir
    
    print("Downloading Pascal VOC dataset...")

    try:
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")

        print("Extracting Pascal VOC dataset...")
        with tarfile.open(filename, "r") as tar:
            tar.extractall(path="./Data")

        os.remove(filename)  # Remove the tar file after extraction
        print("Extraction complete.")
        return pascal_dir
    except Exception as e:
        print(f"An error occurred while downloading or extracting the dataset: {e}")
        return None

def setup_wandb(
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
        threshold=0.5
        ):
    """
    Setup wandb for logging
    """
    wb.init(
        project="PCE",
        entity="your_entity_name",  # Replace with your WandB entity name
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
            'current dataset' : current_dataset
        }
    )  

    return wb.config

def log_gradient_stats(model):
    """Log statistiche sui gradienti"""
    grad_stats = {}
    total_grad_norm = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad.data).item()
            grad_max = torch.max(torch.abs(param.grad.data)).item()
            
            grad_stats[f"grad_norm/{name}"] = grad_norm
            grad_stats[f"grad_max/{name}"] = grad_max
            total_grad_norm += grad_norm ** 2
    
    grad_stats["global/grad_norm"] = (total_grad_norm ** 0.5)
    
    wb.log(grad_stats)

def router_variance_loss(expert_weights, variance_weight = 0.01):
    """
    Calculate the variance loss for the router weights.
    
    Args:
        expert_weights (torch.Tensor): The weights of the experts.
        variance_weight (float): Weight for the variance loss.
        
    Returns:
        torch.Tensor: The variance loss.
    """
    # Calculate the variance of the expert weights
    variance = torch.var(expert_weights, dim=0)
    
    # Return the weighted variance loss
    return variance_weight * torch.sum(variance)

def get_cifar10_sets(batch_size, cifar10_path = './Data/cifar-10-batches-py/data_batch_'):
    """
    Initialize the CIFAR-10 dataset and create DataLoaders for training and validation.

    Args:
        cifar10_path : Path of the CIFAR-10 dataset
    """
    print('-- Initializing the CIFAR-10 dataset... -- ')
    # Create CIFAR-10 dataset
    cifar10 = CIFAR10Dataset(cifar10_path)
    cifar10_train_set = CIFAR10TrainDataset(cifar10)
    cifar10_val_set = CIFAR10ValidationDataset(cifar10)

    # Create DataLoader for CIFAR-10 training and validation datasets
    cifar10_train_dataloader = DataLoader(cifar10_train_set, batch_size=batch_size, shuffle=True)
    cifar10_val_dataloader = DataLoader(cifar10_val_set, batch_size=batch_size, shuffle=False)

    cifar10_dict = {
        'datasets': {
            'train': cifar10_train_set,
            'val': cifar10_val_set,
        },
        'dataloaders': {
            'train': cifar10_train_dataloader,
            'val': cifar10_val_dataloader
        },
        'name' : 'CIFAR10'
    }

    return cifar10_dict

def get_tinyimagenet_sets(batch_size, tinyimagenet_path = '.Data/tiny-imagenet-200'):
    """
    Initialize the Tiny-ImageNet dataset and create Dataloaders for training and validation

    Args:
        tinyimagenet_path : Path of the Tiny-ImageNet dataset
    """

    tiny_image_net = TinyImageNetDataset(tinyimagenet_path)
    tiny_image_net_train_set = TinyImageNetTrainDataset(tiny_image_net)
    tiny_image_net_val_set = TinyImageNetValidationDataset(tiny_image_net)

    tiny_image_net_train_dataloader = DataLoader(dataset=tiny_image_net_train_set, batch_size=batch_size, shuffle=True)
    tiny_image_net_val_dataloader = DataLoader(dataset=tiny_image_net_val_set, batch_size=batch_size, shuffle=False)

    tiny_image_net_dic = {
        'datasets' : {
            'train' : tiny_image_net_train_set,
            'val': tiny_image_net_val_set
        },
        'dataloader' : {
            'train' : tiny_image_net_train_dataloader,
            'val' : tiny_image_net_val_dataloader
        }
    }

    return tiny_image_net_dic

def model_checkpoints(train_checkpoints_path, model):
    # Check if exist model checkpoint
    if os.path.exists(f'{train_checkpoints_path}/model_checkpoints.pth'):
        print(f'Loading model from checkpoint...')
        model.load_state_dict(torch.load(f'{train_checkpoints_path}/model_checkpoints.pth'))

        print('Model loaded from checkpoint')

    else:
        print('No model checkpoint founded')

    return model

def train_checkpoints(train_checkpoints_path, optimizer, lr_scheduler):
    # Check if exist train params checkpoint, included scheduler and optimizer
    if os.path.exists(f'{train_checkpoints_path}/train_checkpoints.pth'):
        print('Load training params from checkpoint...')
        checkpoint = torch.load(f'{train_checkpoints_path}/train_checkpoints.pth')

        start_epoch = checkpoint['start_epoch']
        start_train_batch = checkpoint['train_batch']

        train_loss_history = checkpoint['train_history']
        val_loss_history = checkpoint['val_history']

        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])

        print(f'Resuming train from {start_epoch} epoch and {start_train_batch} train batch')
        
        return start_epoch, start_train_batch, train_loss_history, val_loss_history, optimizer, lr_scheduler
    else:
        print('No train checkpoint founded')

    return 0, 0, [], [], optimizer, lr_scheduler

def save_checkpoint(train_checkpoints_path, model, optimizer, lr_scheduler, 
                   epoch, batch_idx, train_loss_history, val_loss_history):
    """Save model and training state"""

    if not os.path.exists(train_checkpoints_path):
        os.makedirs(train_checkpoints_path)
    
    # Save model
    torch.save(model.state_dict(), f'{train_checkpoints_path}/model_checkpoints.pth')
    
    # Save training state
    torch.save({
        'start_epoch': epoch,
        'train_batch': batch_idx,
        'train_history': train_loss_history,
        'val_history': val_loss_history,
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }, f'{train_checkpoints_path}/train_checkpoints.pth')

def training(
        model, 
        train_loader, 
        val_loader, 
        device,
        epochs,
        optimizer, 
        lr_scheduler,
        augmentation,
        logger,
        train_checkpoints_path = './checkpoints'):
    
    # Setting train and val loader
    train_loader = train_loader
    val_loader = val_loader
    
    # Set loss
    criterion = nn.CrossEntropyLoss()

    # Get train params from the last train checkpoint (if exist)
    start_epoch, start_train_batch, train_loss_history, \
    val_loss_history, optimizer, lr_scheduler = train_checkpoints(
                                                    train_checkpoints_path,
                                                    optimizer,
                                                    lr_scheduler)

    # Get model from the last model checkpoint (if exist)
    model = model_checkpoints(train_checkpoints_path, model)
    model.to(device)

    save_every_n_batches = 5
 
    # Start training
    for epoch in tqdm(range(start_epoch, epochs)):
        model.train()

        train_loss = val_loss = 0
        epoch_train_loss = 0
        batch_count = 0

        # Resume training from last batch save in to checkpoint
        train_batches = list(enumerate(train_loader))
        current_batch_start = start_train_batch if epoch == start_epoch else 0

        if current_batch_start > 0:
            train_batches = train_batches[current_batch_start:]

        for batch_idx, (data, labels) in tqdm(train_batches, desc = 'Training batches'):

            # Extract data and labels from batch
            data, labels = data.to(device), labels.to(device)

            # transform data with data augmentation
            data = augmentation(data)

            # Forward pass
            logits = model(data)

            # Compute loss and backward pass
            loss = criterion(logits, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            lr_scheduler.step()

            # # Track losses
            batch_loss = loss.item()
            epoch_train_loss += batch_loss
            train_loss_history.append(batch_loss)
            batch_count += 1
            
            # Update actual batch idx
            actual_batch_idx = batch_idx if epoch == start_epoch else batch_idx + start_train_batch
            
            if actual_batch_idx % save_every_n_batches == 0:
                save_checkpoint(train_checkpoints_path, model, optimizer, lr_scheduler,
                              epoch, actual_batch_idx, train_loss_history, val_loss_history)
        # Calculate avg train loss
        avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc='Validation batches'):
                data, labels = data.to(device), labels.to(device)
                
                logits = model(data)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        val_loss_history.append(avg_val_loss)
                
        # Save checkpoint at end of epoch
        save_checkpoint(train_checkpoints_path, model, optimizer, lr_scheduler,
                       epoch + 1, 0, train_loss_history, val_loss_history)
        
        # Reset batch counter for next epoch
        start_train_batch = 0

        current_lr = optimizer.param.groups[0]['lr']
        
        if batch_idx % 50 == 0:
            wb.log({
                'current epoch' : epoch,
                'current batch' : batch_idx,
                'batch_loss' : batch_loss,
                'avg train loss' : avg_train_loss,
                'avg val loss' : avg_val_loss,
                'gradient norm ' : log_gradient_stats(model),

            })
    
    print('Training completed!')
    return model, train_loss_history, val_loss_history

if __name__ == "__main__":
    train_datasets = []

    # Hyperparameters of model
    num_exp = 4
    kernel_size = 3
    out_channel_exp = 8
    out_channel_rout = 20
    layer_number = 4
    patch_size = 16
    lr = 0.001
    dropout = 0.1
    weight_decay = 1e-4
    ema_alpha = 0.99
    threshold = 0.2

    epochs = 250
    pre_train_epochs = 150
    fine_tune_epochs = 100
    phase_multipliers = [1.0, 0.3]

    batch_size = 32

    train_config={
        'epochs': epochs,
        'pre_train_epochs': pre_train_epochs,
        'fine_tune_epochs': fine_tune_epochs,
        'epochs' : epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
    }

    # # Create DataLoader for training and validation of all datasets
    # Load Cifar-10 Sets
    cifar10_sets = get_cifar10_sets()
    train_datasets.append(cifar10_sets)

    # Load TinyImageNet sets
    tinyimagenet_sets = get_tinyimagenet_sets()
    train_datasets.append(tinyimagenet_sets)

    patch_esxtractor = PatchExtractor(patch_size=patch_size)

    # Load pascalvoc sets
    # pascalvoc_sets = get_pascalvoc_sets()
    # train_datasets.append(pascalvoc_sets)

     # idx of the dataset : 
        # 0 -> Cifar10
        # 1 -> Tiny-ImageNet
    dataset_idx = 0
        
    # Define dataset and dataloader
    train_dataset = train_datasets[dataset_idx]['datasets']['train']
    
    train_loader = train_datasets[dataset_idx]['dataloader']['train']
    validation_loader = train_datasets[dataset_idx]['dataloader']['val']

    # Setup wandb for logging
    logger = setup_wandb(
        project_name="PCE",
        num_exp=num_exp,
        kernel_size=kernel_size,
        out_channel_exp=out_channel_exp,
        out_channel_rout=out_channel_rout,
        layer_number=layer_number,
        patch_size=patch_size,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        dropout=dropout,
        ema_alpha=ema_alpha,
        weight_decay=weight_decay,
        threshold=threshold,
        current_dataset = train_datasets[dataset_idx]['name']
    )

    # Divides current dataset for initialize keys and setting input channel for model
    dataset_patch, _, _ = patch_esxtractor(train_dataset.data)  # Extract patches from the first 10 images for router initialization
    _, _, C, _, _ = dataset_patch.shape
    
    router = Router(num_experts=num_exp, out_channel_key=out_channel_exp)
    router.initialize_keys(dataset_patch)

    # Initialize model, optimizer (Adam) and scheduler
    # Loss initialize in training funcion (CrossEntropy)
    model = PCENetwork(
        inpt_channel=C,
        num_experts = num_exp,
        kernel_sz_exps = kernel_size,
        output_cha_exps = out_channel_exp,
        layer_number = layer_number,
        patch_size = patch_size,
        router=router,
        dropout=0.1,
        threshold=threshold,
        enable_ema=True
    )
    print('-- Router and model initialized -- ')

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = PCEScheduler(
        optimizer = optimizer, 
        phase_epochs=[pre_train_epochs, fine_tune_epochs],
        base_lr=lr,
        phase_multipliers=phase_multipliers)
    
    augmentation = DataAgumentation()

    training(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        device= 'cuda' if torch.cuda.is_available() else 'cpu',
        epochs=epochs,
        optimizer=optimizer,
        augmentation = augmentation,
        logger=logger,
        lr_scheduler=lr_scheduler,
    )