import torch
import wandb as wb

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import DataLoader

from Datasets_Classes.Cifar10 import CIFAR10Dataset
from Datasets_Classes.PatchExtractor import PatchExtractor

from Model.PCE import PCENetwork
from Model.Components.Router import Router
from PCEScheduler import PCEScheduler
from TrainModel import TrainModel

def setup_wandb(
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
        }
    )  

    return wb.config

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

def split_dataset(dataset, train_ratio = 0.8, val_ratio = 0.2):
    """
    Split dataset into training, validation, and test sets.
    
    Args:
        dataset (Any dataset): The any dataset.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        
    Returns:
        tuple: Training, validation, and test datasets.
    """
    total_size = dataset.__len__()
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return train_set, val_set, test_set

if __name__ == "__main__":
    train_datasets = []

    print('-- Initializing the CIFAR-10 dataset... -- ')
    # Create CIFAR-10 dataset
    cifar10 = CIFAR10Dataset('./Data/cifar-10-batches-py/data_batch_')
    
    # Split cifar10 dataset into train, validation, and test sets
    cifar10_train, cifar10_val, cifar10_test = split_dataset(cifar10)
    cifar10_train_set, cifar10_val_set, cifar10_test_set = cifar10_train.dataset, cifar10_val.dataset, cifar10_test.dataset
    print('-- CIFAR-10 dataset initialized -- ')

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

    epochs = 150
    pre_train_epochs = 100
    fine_tune_epochs = 50

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

    # # Create DataLoader for training and validation
    # # Setup wandb for logging
    # logger = setup_wandb(
    #     project_name="PCE",
    #     num_exp=num_exp,
    #     kernel_size=kernel_size,
    #     out_channel_exp=out_channel_exp,
    #     out_channel_rout=out_channel_rout,
    #     layer_number=layer_number,
    #     patch_size=patch_size,
    #     lr=lr,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     dropout=dropout,
    #     ema_alpha=ema_alpha,
    #     weight_decay=weight_decay,
    #     threshold=threshold
    # )
    print('-- extracting patches from CIFAR-10 dataset... -- ')
    patch_esxtractor = PatchExtractor(patch_size=patch_size)
    print('-- Patches extracted from CIFAR-10 dataset -- ')

    # Create dictionary for CIFAR-10 dataset and append in train_datasets list for training class
    cifar10_dict = {
        'datasets': {
            'train': cifar10_train,
            'val': cifar10_val,
        },
        'dataloaders': {
            'train': DataLoader(cifar10_train, batch_size=batch_size, shuffle=True),
            'val': DataLoader(cifar10_val, batch_size=batch_size, shuffle=False),
        },
        'name' : 'CIFAR10'
    }
    train_datasets.append(cifar10_dict)

    # Initialize the router with the number of experts and initialize model
    # with the router and other parameters
    print('-- Initializing the router and model... -- ')
    fake_dataset_patches, _, _ = patch_esxtractor(cifar10_train_set.data[:10, :, :, :])  # Extract patches from the first 10 images for router initialization
    router = Router(num_experts=num_exp, out_channel_key=out_channel_exp)
    router.initialize_keys(fake_dataset_patches)

    model = PCENetwork(
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
    # # Initialize the train model class
    train_model = TrainModel(
        datasets=train_datasets,
        model=model,
        logger=None,
        config = train_config,
        device = "cuda" if torch.cuda.is_available() else "cpu"
    )

    print('-- Checking train function ... --')
    train_model.train(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_checkpoints_path='./checkpoints',
    )
