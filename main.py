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

if __name__ == "__main__":
    # Create CIFAR-10 dataset
    cifar10 = CIFAR10Dataset('./Data/cifar-10-batches-py/data_batch_')
    print(f'cifar10 dataset : {cifar10.data.shape}, labels : {cifar10.lables.shape}')
    
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

    # Create DataLoader for training and validation
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
        threshold=threshold
    )

    patch_extractor = PatchExtractor(patch_size=patch_size)

    fake_patch_train = torch.rand(10, 3, 128, 128)
    fake_patch_val = torch.rand(5,3,128,128)

    train_set_patches, _, _ = patch_extractor(fake_patch_train)
    
    # Initialize the router with the number of experts and initialize model
    # with the router and other parameters
    router = Router(num_experts=num_exp, out_channel_key=out_channel_exp)
    router.initialize_keys(train_set_patches)

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

    # Define optimizer
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    # Define custom LR scheduler 
    scheduler = PCEScheduler(
        optimizer=optimizer,
        phase_epochs=[pre_train_epochs, fine_tune_epochs],
        base_lr=lr,
        phase_multipliers=[1.0, 0.3],
        last_epoch=-1
    )

    for epoch in range(enumerate(epochs)):
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0

        # Simulate training loop
        for batch in range(100):