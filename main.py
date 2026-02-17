import os
import tarfile
import urllib.request
import numpy as np 
import matplotlib.pyplot as plt
import itertools

import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
gc.collect()
torch.cuda.empty_cache()

from tqdm import tqdm

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from Datasets_Classes.Cifar10 import CIFAR10Dataset, CIFAR10TrainDataset, CIFAR10ValidationDataset
from Datasets_Classes.Cifar100 import CIFAR100Dataset, CIFAR100TrainDataset, CIFAR100ValidationDataset
from Datasets_Classes.TinyImageNet import TinyImageNetDataset, TinyImageNetTrainDataset, TinyImageNetValidationDataset

from Model.PCE import PCENetwork

# from Datasets_Classes.PascalVOC import PascalVOCDataset, PascalVOCTrainDataset, PascalVOCValidationDataset
# from Datasets_Classes.PatchExtractor import PatchExtractor

from EMADiffLitModule import EMADiffLitModule

def count_number_of_classes(train_labels, val_labels):   
    return len(np.unique(np.concatenate([train_labels, val_labels])).tolist())

def get_cifar10_sets(batch_size, cifar10_path = './Data/cifar-10-batches-py'):
    """
    Initialize the CIFAR-10 dataset and create DataLoaders for training and validation.

    Args:
        batch_size (int) 
        cifar10_path (str): Path of the CIFAR-10 dataset
    """
    print('-- Initializing the CIFAR-10 dataset... -- ')
    # Create CIFAR-10 dataset
    cifar10 = CIFAR10Dataset(cifar10_path)
    cifar10_train = CIFAR10TrainDataset(cifar10)
    cifar10_val = CIFAR10ValidationDataset(cifar10)

    # Create DataLoader for CIFAR-10 training and validation datasets
    cifar10_train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    cifar10_val_loader = DataLoader(cifar10_val, batch_size=batch_size, shuffle=False)

    # Get numer of classes in labels
    num_classes = count_number_of_classes(cifar10_train.lables, cifar10_val.lables)

    return {
        'datasets': {
            'train': cifar10_train,
            'val': cifar10_val,
        },
        'dataloaders': {
            'train': cifar10_train_loader,
            'val': cifar10_val_loader
        },
        'name' : 'CIFAR10',
        'num_classes' : num_classes,
    }


def get_tinyimagenet_sets(batch_size, tinyimagenet_path = 'Data/tiny-imagenet-200'):
    """
    Initialize the Tiny-ImageNet dataset and create Dataloaders for training and validation

    Args:
        tinyimagenet_path : Path of the Tiny-ImageNet dataset
    """

    tiny_image_net = TinyImageNetDataset(tinyimagenet_path)
    tiny_image_net_train_set = TinyImageNetTrainDataset(tiny_image_net)
    tiny_image_net_val_set = TinyImageNetValidationDataset(tiny_image_net)

    tiny_image_net_train_dataloader = DataLoader(dataset=tiny_image_net_train_set, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    tiny_image_net_val_dataloader = DataLoader(dataset=tiny_image_net_val_set, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    # Get numer of classes in labels
    all_labeles = np.concatenate([tiny_image_net_train_set.labels, tiny_image_net_val_set.labels])
    unique_labels = np.unique(all_labeles).tolist()
    num_classes = len(unique_labels)

    tiny_image_net_dic = {
        'datasets' : {
            'train' : tiny_image_net_train_set,
            'val': tiny_image_net_val_set
        },
        'dataloaders' : {
            'train' : tiny_image_net_train_dataloader,
            'val' : tiny_image_net_val_dataloader
        },
        'name' : 'TinyImage-Net',
        'num_classes' : num_classes,
        'unique_lables' : unique_labels
    }

    return tiny_image_net_dic

def get_cifar100_sets(batch_size, cifar100_path = './Data/cifar-100-python'):
    """
    Initialize the Cifar-100 dataset and create Dataloaders for training and validation
    Args : 
        batch_size (int) 
        cifar100_path (str): Path of the CIFAR-100 dataset
    """
    cifar100 = CIFAR100Dataset(cifar100_path)
    cifar100_train = CIFAR100TrainDataset(cifar100)
    cifar100_val = CIFAR100ValidationDataset(cifar100)

    cifar100_train_loader = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True)
    cifar100_val_loader = DataLoader(cifar100_val, batch_size=batch_size, shuffle=False)

    num_classes = count_number_of_classes(cifar100_train.lables, cifar100_val.lables)

    return {
        'datasets': {
            'train': cifar100_train,
            'val': cifar100_val,
        },
        'dataloaders': {
            'train': cifar100_train_loader,
            'val': cifar100_val_loader
        },
        'name' : 'CIFAR100',
        'num_classes' : num_classes,
    }

if __name__ == "__main__":

    train_datasets = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'-- Start with device : {device} ---')
    print('\n ------------------------ \n')

    # param_grid = {
    #     'dropout_exp': [0.10, 0.15, 0.20],
    #     'eom_p': [0.10, 0.15, 0.20],
    #     'dropout_head': [0.10, 0.15]
    # }
    # keys, values = zip(*param_grid.items())
    # combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Hyperparameters of model
    num_exp = 16
    layer_number = 8
    patch_size = 16
    lr = 0.0002
    router_lr = 0.0005
    dropout_exp = 0.15
    dropout_head = 0.10
    drop_path = 0.20
    eom_p = 0.15
    weight_decay = 0.004 # M

    # Hyperparameters of router
    capacity_factor_train = 2.0
    capacity_factor_val = 2.25

    alpha_init = 7e-3
    alpha_final = 4e-4 # M
    alpha_epochs =  200

    temp_init = 2.0
    temp_mid = 1.2
    temp_final = 0.65
    temp_epochs = 200

    # Training metrics
    train_epochs = 150
    uniform_epochs = 35
    batch_size = 128

    print("\n--- Hyperparameters ---")
    print(f"Model: experts={num_exp},layers={layer_number}, patch={patch_size}, lr={lr}, wd={weight_decay}")
    print(f"Training: epochs={train_epochs}, batch={batch_size}\n")
    print('\n ------------------------ \n')

    # cifar10_sets = get_cifar10_sets(batch_size)
    # cifar100_sets = get_cifar100_sets(batch_size)

    tiny_set = get_tinyimagenet_sets(batch_size)
    train_loader = tiny_set['dataloaders']['train']
    val_loader = tiny_set['dataloaders']['val']
    num_classes = tiny_set['num_classes']


    print(f'Num classes : {num_classes}')

    print(f'--- Dataset loaded --- \n')

    # for i, params in tqdm(enumerate(combinations)):
        # dropout_exp = params['dropout_exp']
        # eom_p = params['eom_p']
        # dropout_head = params['dropout_head']

    run_name = f"test-CutMix-MixAlpha"

    # Defines checkpointer and Logger
    logger = WandbLogger(
        project="PCE",
        log_model = False,
        name = f'Test-Tiny-{run_name}',
    )
    logger.experiment.define_metric("epoch")
    logger.experiment.define_metric("*", step_metric="epoch")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        filename = 'best-model',
        dirpath = 'checkpoints/',
        save_weights_only = False,
    )
    pce = PCENetwork(
        num_experts = num_exp,
        layer_number = layer_number,
        patch_size = patch_size,
        dropout_exp = dropout_exp,
        dropout_head = dropout_head,
        drop_path = drop_path,
        num_classes=num_classes,
        router_temp=temp_init,
        capacity_factor_train = capacity_factor_train,
        capacity_factor_val = capacity_factor_val,
        eom_p = eom_p
        )
    # pce = torch.compile(pce, mode="reduce-overhead")

    lit_module = EMADiffLitModule(
        pce=pce, lr=lr, weight_decay=weight_decay, device=device, train_epochs=train_epochs, 
        uniform_epochs=uniform_epochs, alpha_init=alpha_init,  alpha_final=alpha_final, 
        alpha_epochs=alpha_epochs, temp_init=temp_init, temp_mid = temp_mid, 
        temp_final=temp_final,temp_epochs=temp_epochs, num_classes=num_classes, router_lr = router_lr
    )
    trainer = pl.Trainer(
        max_epochs=train_epochs,
        logger = logger,
        precision='32',
        accelerator=device,
        enable_checkpointing= True,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
    )

    print(f'--- Start training --- \n')
    if os.path.exists('checkpoints/last.ckpt'):
        trainer.fit(lit_module, train_loader, val_loader, ckpt_path='checkpoints/last.ckpt')
    else:
        trainer.fit(lit_module, train_loader, val_loader)

    logger.experiment.finish()

    del pce, lit_module, trainer, logger
    torch.cuda.empty_cache()
    gc.collect()       