import os
# os.environ['WANDB_MODE'] = 'offline'
import io
import tarfile
import urllib.request
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import pytorch_lightning as pl

from DataAugmentation import DataAgumentation

from Datasets_Classes.Cifar10 import CIFAR10Dataset, CIFAR10TrainDataset, CIFAR10ValidationDataset
from Datasets_Classes.TinyImageNet import TinyImageNetDataset, TinyImageNetTrainDataset, TinyImageNetValidationDataset
# from Datasets_Classes.PascalVOC import PascalVOCDataset, PascalVOCTrainDataset, PascalVOCValidationDataset
from Datasets_Classes.PatchExtractor import PatchExtractor

from Model.PCE import PCENetwork

from Training import Checkpointer
from Training import Logger
from Training.BackBone_Trainer import BackboneCheckpointCallBack, BackboneLitModule, BackboneLoggerCallBack

from PCEScheduler import PCEScheduler

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

def training(logger, checkpointer, PCE, optimizer, lr_scheduler, num_classes, str_epoch, str_train_batch, 
                   train_loss_history , val_loss_history, pre_train_epochs, fine_tune_epochs, phase_multipliers, weight_decay):
    
    if 0 <= str_epoch <= back_bone_epochs:
        logger_cb = BackboneLoggerCallBack(logger = logger, backbone_epochs = back_bone_epochs, log_predicttion_every_batch = 5)
        checkpointer_cb = BackboneCheckpointCallBack(checkpointer, 5)
        backbone_lit_module = BackboneLitModule(PCE, 
                                                optimizer, 
                                                lr_scheduler, 
                                                num_classes, 
                                                str_epoch, 
                                                str_train_batch,
                                                train_loss_history, 
                                                val_loss_history,
                                                pre_train_epochs,
                                                fine_tune_epochs,
                                                phase_multipliers,
                                                back_bone_epochs
                                            )

        trainer = pl.Trainer(
            max_epochs = back_bone_epochs,
            logger = False,
            callbacks = [logger_cb, checkpointer_cb]
        )
    elif back_bone_epochs < str_epoch <= ema_only_epochs:
        pass

if __name__ == "__main__":

    train_datasets = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'-- Start with device : {device} ---')
    print('\n ------------------------ \n')

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

    # Hyperparameters of router
    ema_alpha = 0.99
    router_temperature = 1.0
    threshold = 0.2
    hard_threshold_router = False

    # Training metrics
    back_bone_epochs = 50
    ema_only_epochs = 100
    differentiable_epochs = 100
    total_epoch = back_bone_epochs + ema_only_epochs + differentiable_epochs
    phase_multipliers = [1.0, 0.3]
    batch_size = 32

    print("\n--- Hyperparameters ---")
    print(f"Model: experts={num_exp}, k={kernel_size}, out_exp={out_channel_exp}, out_rout={out_channel_rout}, layers={layer_number}, patch={patch_size}, lr={lr}, dropout={dropout}, wd={weight_decay}")
    print(f"Router: ema={ema_alpha}, temp={router_temperature}, thresh={threshold}, hard={hard_threshold_router}")
    print(f"Training: epochs={total_epoch} (Backbone epochs={back_bone_epochs}, ema only epochs={ema_only_epochs}), differentiable epochs = {differentiable_epochs}  \
        phases={phase_multipliers}, batch={batch_size}\n")
    print('\n ------------------------ \n')

    # train_config={
    #     'epochs': epochs,
    #     'pre_train_epochs': pre_train_epochs,
    #     'fine_tune_epochs': fine_tune_epochs,
    #     'epochs' : epochs,
    #     'batch_size': batch_size,
    #     'lr': lr,
    #     'weight_decay': weight_decay,
    # }

    # Initialize patch extractor
    patch_extractor = PatchExtractor(patch_size=patch_size)


    # # Create DataLoader for training and validation of all datasets
    # Load Cifar-10 Sets
    cifar10_sets = get_cifar10_sets(batch_size)
    train_datasets.append(cifar10_sets)

    # Load TinyImageNet sets
    # tinyimagenet_sets = get_tinyimagenet_sets(batch_size)
    # train_datasets.append(tinyimagenet_sets)

    # Load pascalvoc sets
    # pascalvoc_sets = get_pascalvoc_sets()
    # train_datasets.append(pascalvoc_sets)

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

    print(f'--- Dataset loaded --- \n')

    print('--- Loading router and model ---')
    # Divides current dataset in patch for initialize keys and setting input channel for model
    dataset_patch, _, _ = patch_extractor(train_dataset.data)  # Extract patches 
    _, _, C, _, _ = dataset_patch.shape

    # Initialize model, optimizer (Adam) and scheduler
    # Loss initialize in training funcion (CrossEntropy)
    PCE = PCENetwork(
        inpt_channel= C,
        num_experts = num_exp,
        layer_number = layer_number,
        patch_size = patch_size,
        dropout=0.1,
        num_classes=num_classes,
        enable_router_metrics=True,
        hard_threshold_router = False,
    )
    
    print('-- model initialized -- \n')    
    augmentation = DataAgumentation()

    # Defines checkpointer and Logger
    checkpointer = Checkpointer('./checkpoints')
    logger = Logger()
    logger.setup_wandb(
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
        router_temperature = router_temperature,
        current_dataset = 'Fake dataset'
    )

    str_epoch = checkpointer.get_start_epoch()

    #Phase 1 :  Backbone training phase
    if str_epoch == 0:
        # Freeze router
        for p in PCE.router.parameters() : p.requires_grad = False
        optim_params = list(filter(lambda p : p.requires_grad, PCE.parameters()))

        optimizer = Adam(optim_params, lr=lr, weight_decay=weight_decay)
        lr_scheduler = PCEScheduler(
            optimizer= optimizer,
            phase_epochs=[back_bone_epochs + ema_only_epochs, differentiable_epochs],
            base_lr=lr,
            phase_multipliers=phase_multipliers
        )
    else:
        str_epoch, str_train_batch, train_loss_history, \
        val_loss_history, optimizer, lr_scheduler = checkpointer.train_checkpoints()

    # Phase 2 : Unfreeze + add router params (keys undifferentiable)
    if back_bone_epochs < str_epoch <= ema_only_epochs:
        for p in PCE.router.parameters() : p.requires_grad = True
        for key in PCE.router.keys: key.requires_grad = False

        if str_epoch == back_bone_epochs + 1:
            new_params = [p for p in PCE.router.parameters() if p.requires_grad]
            optimizer.add_param_group(
                {'params' : new_params}
            )

    # Phase 3 : full differentiable
    if ema_only_epochs < str_epoch <= differentiable_epochs:
        for key in PCE.router.keys: key.requires_grad = True

        if str_epoch == ema_only_epochs + 1 :
            new_params = [p for p in PCE.router.parameters() if p.requires_grad]
            optimizer.add_param_group(
                {'params' : new_params}
            )
    
    str_epoch, str_train_batch, train_loss_history, \
        val_loss_history, optimizer, lr_scheduler = checkpointer.train_checkpoints()
    
    training(logger, checkpointer, PCE, optimizer, lr_scheduler, num_classes, str_epoch, str_train_batch, 
             train_loss_history, val_loss_history, total_epoch, back_bone_epochs, 
             ema_only_epochs, differentiable_epochs, weight_decay)