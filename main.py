import torch
import wandb as wb

import os
import tarfile
import requests
import zipfile
import shutil
import urllib.request
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import DataLoader

from zipfile import ZipFile

from Datasets_Classes.Cifar10 import CIFAR10Dataset, CIFAR10TrainDataset, CIFAR10ValidationDataset
from Datasets_Classes.TinyImageNet import TinyImageNetDataset, TinyImageNetTrainDataset, TinyImageNetValidationDataset
# from Datasets_Classes.PascalVOC import PascalVOCDataset, PascalVOCTrainDataset, PascalVOCValidationDataset
from Datasets_Classes.PatchExtractor import PatchExtractor

from Model.PCE import PCENetwork
from Model.Components.Router import Router

from PCEScheduler import PCEScheduler
from TrainModel import TrainModel

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

    for dataset_idx, dataset in enumerate(train_datasets):
        
        # Define dataset and dataloader
        train_dataset = dataset['datasets']['train']
        
        train_loader = dataset['dataloader']['train']
        validation_loader = dataset['dataloader']['val']

        # Divides current dataset for initialize keys and setting input channel for model
        dataset_patch, _, _ = patch_esxtractor(train_dataset.data)  # Extract patches from the first 10 images for router initialization
        _, _, C, _, _ = dataset_patch.shape
        
        router = Router(num_experts=num_exp, out_channel_key=out_channel_exp)
        router.initialize_keys(dataset_patch)

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

        # # Initialize the train model class
        train_model = TrainModel(
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
