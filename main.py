import os
# os.environ['WANDB_MODE'] = 'offline'
import io
import tarfile
import urllib.request
import numpy as np 
import matplotlib.pyplot as plt

import torch

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from DataAugmentation import DataAgumentation

from Datasets_Classes.Cifar10 import CIFAR10Dataset, CIFAR10TrainDataset, CIFAR10ValidationDataset
from Datasets_Classes.TinyImageNet import TinyImageNetDataset, TinyImageNetTrainDataset, TinyImageNetValidationDataset
# from Datasets_Classes.PascalVOC import PascalVOCDataset, PascalVOCTrainDataset, PascalVOCValidationDataset
from Datasets_Classes.PatchExtractor import PatchExtractor

from Model.PCE import PCENetwork

# from Training.Checkpointer import Checkpointer
# from Training.Logger import Logger
# from Training.CheckpointCallBack import CheckpointCallBack


# from Training.EMA_Diff_Trainer.EMADiffLitModule import EMADiffLitModule
# from Training.EMA_Diff_Trainer.EMADiffLoggerCallBack import EMADiffLoggerCallBack

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

    # Get numer of classes in labels
    all_labeles = np.concatenate([cifar10_train_set.lables, cifar10_val_set.lables])
    unique_labels = np.unique(all_labeles).tolist()
    num_classes = len(unique_labels)

    cifar10_dict = {
        'datasets': {
            'train': cifar10_train_set,
            'val': cifar10_val_set,
        },
        'dataloaders': {
            'train': cifar10_train_dataloader,
            'val': cifar10_val_dataloader
        },
        'name' : 'CIFAR10',
        'num_classes' : num_classes,
        'unique_lables' : unique_labels
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

    # Get numer of classes in labels
    all_labeles = np.concatenate([tiny_image_net_train_set.labels, tiny_image_net_val_set.labels])
    unique_labels = np.unique(all_labeles).tolist()
    num_classes = len(unique_labels)

    tiny_image_net_dic = {
        'datasets' : {
            'train' : tiny_image_net_train_set,
            'val': tiny_image_net_val_set
        },
        'dataloader' : {
            'train' : tiny_image_net_train_dataloader,
            'val' : tiny_image_net_val_dataloader
        },
        'name' : 'TinyImage-Net',
        'num_classes' : num_classes,
        'unique_lables' : unique_labels
    }

    return tiny_image_net_dic

if __name__ == "__main__":

    train_datasets = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'-- Start with device : {device} ---')
    print('\n ------------------------ \n')

    # Hyperparameters of model
    num_exp = 10
    layer_number = 5
    patch_size = 16
    lr = 0.0005
    dropout = 0.25
    weight_decay = 2e-4
    aux_loss_weight = 0.2
    hidden_size = 256

    # Hyperparameters of router
    router_temp = 0.25
    load_factor = 0.02
    noise_epsilon = 1e-2
    capcity_factor_train = 1.25
    capcity_factor_eval = 1.50

    # Training metrics
    train_epochs = 200
    batch_size = 32

    print("\n--- Hyperparameters ---")
    print(f"Model: experts={num_exp},layers={layer_number}, patch={patch_size}, lr={lr}, dropout={dropout}, wd={weight_decay}")
    print(f"Router: temp={router_temp}, load_factor={load_factor}, noise_epsilon={noise_epsilon}, capcity_factor_train={capcity_factor_train}, capcity_factor_eval={capcity_factor_eval}")
    print(f"Training: epochs={train_epochs}, batch={batch_size}\n")
    print('\n ------------------------ \n')

    # Create DataLoader for training and validation of all datasets
    # Load Cifar-10 Sets
    cifar10_sets = get_cifar10_sets(batch_size)
    train_datasets.append(cifar10_sets)

    # Load TinyImageNet sets
    tinyimagenet_sets = get_tinyimagenet_sets(batch_size)
    train_datasets.append(tinyimagenet_sets)  

    # Load pascalvoc sets
    pascalvoc_sets = get_pascalvoc_sets()
    train_datasets.append(pascalvoc_sets)

    #  idx of the dataset : 
    #     0 -> Cifar10
    #     1 -> Tiny-ImageNet
    dataset_idx = 0
        
    # Define dataset and dataloader
    train_dataset = train_datasets[dataset_idx]['datasets']['train']
    
    train_loader = train_datasets[dataset_idx]['dataloaders']['train']
    validation_loader = train_datasets[dataset_idx]['dataloaders']['val']
    num_classes = train_datasets[dataset_idx]['num_classes']
    class_names = train_datasets[dataset_idx]['unique_lables']
    augmentation = DataAgumentation()

    # Defines checkpointer and Logger
    logger = WandbLogger(
        project="PCE",
        log_model = True,
        name = 'Test Warmaup 5 (Top-p)'
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        filename = 'best-model',
        dirpath = 'checkpoints/',
        save_weights_only = True,
    )

    lit_module = EMADiffLitModule(
        num_experts,
        layer_number,
        patch_size,
        dropout,
        num_classes,
        hidden_size,
        lr, 
        weight_decay,
        augmentation,
        class_names,
        device,
        train_epochs,
        aux_loss_weight,
        router_temp,
        load_factor,
        noise_epsilon,
        capcity_factor_train,
        capcity_factor_eval,
    )
    trainer = pl.Trainer(
        max_epochs=train_epochs,
        logger = logger,
        precision='16-mixed',
        gradient_clip_val=0.5,
        gradient_clip_algorithm='norm',
        accelerator=device,
        enable_checkpointing= False,
        callbacks=[checkpoint_callback],
    )
    print(f'--- Start training --- \n')
    if os.path.exists('checkpoints/last.ckpt'):
        trainer.fit(lit_module, train_loader, val_loader, ckpt_path='checkpoints/last.ckpt')
    else:
        trainer.fit(lit_module, train_loader, val_loader)