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

from Training.Checkpointer import Checkpointer
from Training.Logger import Logger
from Training.CheckpointCallBack import CheckpointCallBack


from Training.EMA_Diff_Trainer.EMADiffLitModule import EMADiffLitModule
from Training.EMA_Diff_Trainer.EMADiffLoggerCallBack import EMADiffLoggerCallBack

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

def get_trainable_params(PCE, phase):
    """
    Get trainable params of PCE networks 

    Args : 
        PCE : (nn.Module) is a models
        phase (string) : string that rappresents current phase
    """
    if phase == 'ema_only':
        for p in PCE.router.parameters() : p.requires_grad = True
        for key in PCE.router.keys : key.requires_grad = False
    elif phase == 'deff':
        for key in PCE.router.keys : key.requires_grad = True

def load_checkpoints(
        checkpointer, PCE, lr, weight_decay, phase_multipliers,
        ema_only_epochs, differentiable_epochs):
    """
    Get the last checkpoint if exist

    Args:
        checkpointer (Object): Manager of checkpoint.
        PCE (nn.Module): The model to load the checkpoint into.
        lr (float): Learning rate to use for optimizer.
        weight_decay (float): Weight decay (L2 penalty) for optimizer.
        phase_multipliers (list): Multipliers for different training phases.
        ema_only_epochs (int): Number of epochs for EMA-only training phase.
        differentiable_epochs (int): Number of epochs for differentiable training phase.
    Returns:
        The loaded checkpoint or None if no checkpoint exists.
    """
        
    checkpoint = checkpointer.train_checkpoints()

    if checkpoint is None:
        phase = 'ema_only'
        str_val_batch = str_train_batch = str_epoch = 0
        val_loss_history = train_loss_history = []
        optimizer = Adam(
            params=PCE.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        lr_scheduler = PCEScheduler(
            optimizer=optimizer,
            phase_epochs=[ema_only_epochs, differentiable_epochs],
            base_lr=lr,
            phase_multipliers=phase_multipliers
        )
    else:
        phase = checkpoint['phase']
        str_epoch = checkpoint['start_epoch']
        str_train_batch = checkpoint['train_batch']
        str_val_batch = checkpoint['val_batch']
        train_loss_history = checkpoint['train_history']
        val_loss_history = checkpoint['val_history']

        optimizer = Adam(
            params=PCE.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        lr_scheduler = PCEScheduler(
            optimizer=optimizer,
            phase_epochs=[ema_only_epochs, differentiable_epochs],
            base_lr=lr,
            phase_multipliers=phase_multipliers
        )

        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    
    return phase, str_epoch, str_train_batch, str_val_batch, train_loss_history, val_loss_history, optimizer, lr_scheduler

def training(logger, checkpointer, PCE, val_loader, train_loader, train_set,
             ema_only_epochs, differentiable_epochs, 
             lr, weight_decay, phase_multipliers, device, augmentation, class_names):
    """
    Training function

    Args:
        logger (Logger): Logger object for experiment tracking.
        checkpointer (Checkpointer): Object to manage checkpoints.
        PCE (nn.Module): The model to be trained.
        val_loader (DataLoader): DataLoader for validation data.
        train_loader (DataLoader): DataLoader for training data.
        train_set (Dataset): Training dataset.
        back_bone_epochs (int): Number of epochs for backbone training phase.
        ema_only_epochs (int): Number of epochs for EMA-only training phase.
        differentiable_epochs (int): Number of epochs for differentiable training phase.
        lr (float): Learning rate for optimizer.
        weight_decay (float): Weight decay (L2 penalty) for optimizer.
        phase_multipliers (list): Multipliers for different training phases.
        device (str): Device to use for training ('cuda' or 'cpu').
        augmentation (DataAgumentation): Data augmentation object.
    """

    if isinstance(train_set, np.ndarray):
        train_set = torch.tensor(train_set, dtype=torch.float32)
    
    phases = ['ema_only', 'diff']

    # Get all checkpoints if exists
    PCE = checkpointer.model_checkpoints(PCE)
    str_phase, str_epoch, str_train_batch, str_val_batch, train_loss_history, \
    val_loss_history, optimizer, lr_scheduler = \
    load_checkpoints(checkpointer, PCE, lr, weight_decay, phase_multipliers, ema_only_epochs, differentiable_epochs)

    idx_last_phase = idx_str_phase = phases.index(str_phase)

    checkpointer_cb = CheckpointCallBack(checkpointer, str_epoch, 1)

    get_trainable_params(PCE, str_phase)

    # Phases training
    for phase in range(idx_str_phase, len(phases)):
        actual_phase = phases[phase]
        idx_actual_phase = phases.index(actual_phase)

        if idx_actual_phase > idx_last_phase:
            get_trainable_params(PCE, phase)
        
        if actual_phase == 'ema_only':
            if str_epoch == 0:
                print('-- START Training | Initialize keys ---')
                PCE.initialize_keys(train_set[:10000])
            else: 
                print(f'--- RESUME Training from {str_epoch} epoch')
            logger_cb = EMADiffLoggerCallBack(logger, str_epoch, 5)
            lit_module = EMADiffLitModule(
                PCE,
                lr_scheduler,
                optimizer,
                str_epoch,
                str_train_batch,
                str_val_batch,
                train_loss_history,
                val_loss_history,
                augmentation,
                phase_multipliers,
                lr,
                weight_decay,
                actual_phase,
                class_names,
                device
            )
            trainer = pl.Trainer(
                max_epochs=ema_only_epochs,
                logger = False,
                callbacks=[logger_cb, checkpointer_cb],
                precision='16-mixed',
                gradient_clip_val=0.5,
                gradient_clip_algorithm='norm',
                accelerator=device,
                num_sanity_val_steps=0,
                enable_checkpointing= False,
            )
        if actual_phase == 'diff':
            if str_phase != 'diff':
                str_epoch = ema_only_epochs + 1

            logger_cb = EMADiffLoggerCallBack(logger, str_epoch, 5)
            lit_module = EMADiffLitModule(
                PCE,
                lr_scheduler,
                optimizer,
                str_epoch,
                str_train_batch,
                str_val_batch,
                train_loss_history,
                val_loss_history,
                augmentation,
                phase_multipliers,
                lr,
                weight_decay,
                actual_phase,
                class_names,
                device
            )
            trainer = pl.Trainer(
                max_epochs=differentiable_epochs + ema_only_epochs,
                logger = False,
                callbacks=[logger_cb, checkpointer_cb],
                precision='16-mixed',
                gradient_clip_val=0.5,
                gradient_clip_algorithm='norm',
                accelerator=device,
                num_sanity_val_steps=0,
                enable_checkpointing= False,
            )
        
        trainer.fit(lit_module, train_loader, val_loader)
        idx_last_phase = idx_actual_phase

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
    ema_only_epochs = 100
    differentiable_epochs = 100
    total_epoch = ema_only_epochs + differentiable_epochs
    phase_multipliers = [1.0, 0.3]
    batch_size = 32

    print("\n--- Hyperparameters ---")
    print(f"Model: experts={num_exp}, k={kernel_size}, out_exp={out_channel_exp}, out_rout={out_channel_rout}, layers={layer_number}, patch={patch_size}, lr={lr}, dropout={dropout}, wd={weight_decay}")
    print(f"Router: ema={ema_alpha}, temp={router_temperature}, thresh={threshold}, hard={hard_threshold_router}")
    print(f"Training: epochs={total_epoch} ema only epochs={ema_only_epochs}), differentiable epochs = {differentiable_epochs}  \
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

    # Initialize model, optimizer (Adam) and scheduler
    # Loss initialize in training funcion (CrossEntropy)
    PCE = PCENetwork(
        num_experts = num_exp,
        layer_number = layer_number,
        patch_size = patch_size,
        dropout=0.1,
        num_classes=num_classes,
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
        epochs=total_epoch,
        dropout=dropout,
        ema_alpha=ema_alpha,
        weight_decay=weight_decay,
        threshold=threshold,
        router_temperature = router_temperature,
        current_dataset = 'Fake dataset'
    )

    training(
        logger = logger,
        checkpointer=checkpointer,
        PCE=PCE,
        val_loader=validation_loader,
        train_loader=train_loader,
        train_set=train_dataset.data,
        ema_only_epochs=ema_only_epochs,
        differentiable_epochs=differentiable_epochs,
        lr=lr,
        weight_decay=weight_decay,
        phase_multipliers=phase_multipliers,
        device=device,
        augmentation=augmentation,
        class_names = class_names
    )