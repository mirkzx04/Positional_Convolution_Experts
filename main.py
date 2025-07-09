import wandb as wb
import os
os.environ['WANDB_MODE'] = 'offline'
import io
import tarfile
import urllib.request
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from contextlib import nullcontext
from functools import partial

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from torchvision.transforms import ToPILImage

from torchmetrics import Accuracy

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
        threshold=0.5,
        router_temperature = 0.5
        ):
    """
    Setup wandb for logging
    """
    wb.init(
        project="PCE",
        entity="mirkzx",  # Replace with your WandB entity name
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
            'current dataset' : current_dataset,
            'router_temperature' : router_temperature
        }
    )  

    return wb.config

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

def calc_router_loss(
        model, 
        confidence_weight = 0.01, 
        anticollapse = 0.001, 
        threshold_weight = 0.005,
        return_stats = False
        ):
    """
    Encourage confident expert selection while preventing expert collapse.

    - Confidence : When an expert is chosen, it should be chosen strongly
    - Anti-Collapse : Ensure all experts get used across the dataset over time
    """

    metrics = model.router.get_cached_metrics()
    if metrics is None:
        return torch.tensor(0.0, requires_grad=True)
    
    # Get experts weights [B * P, num_experts]
    expert_weights = metrics['weights_filtered']

    # Confidence loss : Encourage sharp distributions per patch
    # We use entropy : Highet entropy = Less confident, Low entropy = more confident
    patch_entropies = - (expert_weights * torch.log(expert_weights + 1e-8)).sum(dim = 1)
    confidence_loss = patch_entropies.mean()

    # Anti collapse loss
    experts_usage = (expert_weights > 0).float().mean(dim = 0)
    experts_unused = (experts_usage < 0.01).float().sum()
    collapse_loss = experts_unused 

    # Penalize conservative or permissive threshold
    adaptive_threshold = metrics['adaptive_threshold']
    threshold_extreme_penalty = torch.relu(adaptive_threshold - 0.7).mean() + \
                                torch.relu(0.05 - adaptive_threshold).mean()

    total_loss = (confidence_weight * confidence_loss) + \
                (anticollapse * collapse_loss) + \
                (threshold_weight * threshold_extreme_penalty)

    if return_stats:
        router_loss_metrics = {
            'router_loss/confidence_loss': confidence_loss.item(),
            'router_loss/collapse_loss': collapse_loss.item(),
            'router_loss/total_specialization_loss': total_loss.item(),
            'router_loss/avg_patch_entropy': patch_entropies.mean().item(),
            'router_loss/unused_experts_count': experts_unused.item(),
            'router_loss/min_expert_usage': experts_usage.min().item(),
            'router_loss/max_expert_usage': experts_usage.max().item(),

            'router_loss/threshold_extreme_penalty': threshold_extreme_penalty.item(),
            'router_loss/avg_adaptive_threshold': adaptive_threshold.mean().item(),
            'router_loss/threshold_std': adaptive_threshold.std().item(),
            'router_loss/threshold_range': (adaptive_threshold.max() - adaptive_threshold.min()).item(),
        }

        return total_loss, router_loss_metrics

    return total_loss

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
    all_labeles = cifar10.labels_train + cifar10.labels_validation
    unique_labels = list(set(all_labeles))
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
    all_labeles = tiny_image_net.labels_train + tiny_image_net.labels_validation
    unique_labels = list(set(all_labeles))
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

def setup_torchmetrics_accuracy(num_classes, device):
    """
    Setup Top-1 and Top-5 with torchmetrics

    Args:
        num_classes(int) : Number of classes in to dataset
        device : Device (cuda / cpu)
    """

    metrics = {
        'top1_train' : Accuracy(task = 'multiclass', num_classes=num_classes, top_k=1).to(device),
        'top1_val' : Accuracy(task = 'multiclass', num_classes=num_classes, top_k=1).to(device),

        'top5_train' : Accuracy(task = 'multiclass', num_classes=num_classes, top_k=5).to(device),
        'top5_val' : Accuracy(task = 'multiclass', num_classes=num_classes, top_k=5).to(device),
    }

    return metrics

def log_prediction_to_wandb(data_batch, 
                            true_labels,
                            pred_labels, 
                            batch_class_loss,
                            batch_router_loss,
                            batch_total_loss,
                            class_names = None, 
                            num_images_to_log = 10, 
                            epoch = None, 
                            phase = 'train'):
    """
    Log model predictions to wandb with images, true labels and prediction labels

    Args:
        data_batch (torch.Tensor): Batch of images with shape (B, C, H, W)
        true_labels  // : True labels with shape (B,)
        pred_labels // : Predicted labels with shape (B,) or (B, num_classes)
        class_names ( list) : List of class name for CIFAR-10/other datasets
        num_images (int) : Number of images to log
        epoch // : Current epoch number
        step // : Current step number
        phase (str) : Training phase ('Training' or 'Validation')
        batch_loss (float) : loss of the current batch
    """

    # Convert tensors to numpy if needed
    if isinstance(data_batch, torch.Tensor):
        data_batch = data_batch.detach().cpu()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().cpu()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.detach().cpu()

    if len(pred_labels.shape) > 1:
        pred_labels = torch.argmax(pred_labels, dim = 1)
    
    # Limit number of images to log
    num_images_to_log = min(num_images_to_log, data_batch.shape[0])
    wandb_images = []

    for i in range(num_images_to_log):
        # Get singles image and labels
        img = data_batch[i]
        true_label = true_labels[i]
        pred_label = pred_labels[i]

        # Conver images tensor to PIL Image, handle different image formats (CHW vs HWC)
        if img.shape[0] == 3: 
            img_pil = ToPILImage()(img)
        else: #HWC format, transpose to CHW
            img = img.permute(2, 0, 1)
            img_pil = ToPILImage()(img)

        # Create caption with true and predicted labels
        true_class = class_names[true_label] if true_label < len(class_names) else f'True classes : {true_label}'
        pred_class = class_names[pred_label] if pred_label < len(class_names) else f'Predicted classes : {pred_label}'

        is_correct = "✓" if true_label == pred_label else "✗"
        caption = f'{is_correct} True : {true_class} | Pred : {pred_class} \n \
                    loss metrics : batch classification loss : {batch_class_loss} ||  \
                    batch router loss : {batch_router_loss} || batch total loss : {batch_total_loss}'

        # Create wandb image object
        wandb_img = wb.Image(
            img_pil,
            caption=caption
        )

        wandb_images.append(wandb_img)
    
    # Log to wandb
    log_dict = {f'{phase}_predictions' : wandb_images}
    wb.log(log_dict, step = epoch)

def log_train_metrics_to_wandb(
    avg_train_class_loss, avg_train_router_loss, avg_train_total_loss, train_top1_acc,
    train_top5_acc, avg_val_classification_loss, avg_val_router_loss, 
    avg_val_total_loss, val_top1_acc, val_top5_acc, lr, epoch, gradient_norm, 
    best_val_loss, router_metrics
    ):
    """
    Log train metrics to wandb
    
    avg_train_class_loss=avg_train_class_loss,
            avg_train_router_loss=avg_train_router_loss,
            avg_train_total_loss=avg_train_total_loss,
            train_top1_acc=train_top1_acc,
            train_top5_acc = train_top5_acc,
            avg_val_classification_loss=avg_val_classification_loss,
            avg_val_router_loss=avg_val_router_loss,
            avg_val_total_loss=avg_val_total_loss,
            val_top1_acc=val_top1_acc,
            val_top5_acc = val_top5_acc,
            lr=lr,
            epoch=epoch,
            gradient_norm=gradient_norm,
            best_val_loss=best_val_loss,
            router_metrics = router_metrics

    Args:
    avg_train_class_loss (float): Average training classification loss
    avg_train_router_loss (float): Average training router loss
    avg_train_total_loss (float): Average training total loss
    train_accuracy (float): Training accuracy
    avg_val_classification_loss (float): Average validation classification loss
    avg_val_router_loss (float): Average validation router loss
    avg_val_total_loss (float): Average validation total loss
    val_accuracy (float): Validation accuracy
    lr (float): Current learning rate
    epoch (int): Current training epoch
    gradient_norm (float): Norm of gradient
    best_val_loss (float): Best validation loss
    """

    log_dict = {
        'avg_train_class_loss': avg_train_class_loss,
        'avg_train_router_loss': avg_train_router_loss,
        'avg_train_total_loss': avg_train_total_loss,
        'train_top1_acc': train_top1_acc,
        'train_top5_acc' : train_top5_acc,
        'avg_val_classification_loss': avg_val_classification_loss,
        'avg_val_router_loss': avg_val_router_loss,
        'avg_val_total_loss': avg_val_total_loss,
        'val_top1_acc': val_top1_acc,
        'val_top5_acc' : val_top5_acc,
        'lr': lr,
        'epoch': epoch,
        'gradient_norm': gradient_norm,
        'best_val_loss': best_val_loss,
    }

    # Adding router metrics if avaible
    if router_metrics:
        log_dict.update(router_metrics)
    
    wb.log(log_dict, step=epoch)

def training(
        model, 
        train_loader, 
        val_loader, 
        device,
        epochs,
        optimizer, 
        lr_scheduler,
        augmentation,
        accuracy_metrics,
        class_names,
        num_classes,
        train_checkpoints_path = './checkpoints'):
    
    # Setting train and val loader
    train_loader = train_loader
    val_loader = val_loader 

    # Set loss
    criterion = nn.CrossEntropyLoss()

    # Get train params from the last train checkpoint (if exist)
    print('--- Check train checkpoints ---')
    start_epoch, start_train_batch, train_loss_history, \
    val_loss_history, optimizer, lr_scheduler = train_checkpoints(
                                                    train_checkpoints_path,
                                                    optimizer,
                                                    lr_scheduler)
    if start_epoch != 0:
        print(f'Resume training from epoch number : {start_epoch}')
    if start_train_batch != 0:
        print(f'Resume batches from batch_idx : {start_train_batch}')
    print('\n ------------------------ \n')

    # Get model from the last model checkpoint (if exist)
    print('--- Verify model checkpoints ---')
    model = model_checkpoints(train_checkpoints_path, model)
    model.to(device)
    print('\n ------------------------ \n')

    # Setting how often to do training, model and router logging
    save_checkpoint_every = 5
    log_prediction_every = 100

    # Print checkpoint and logging intervals
    print(f"Checkpoint will be saved every {save_checkpoint_every} batches.")
    print(f"Predictions will be logged every {log_prediction_every} batches.")
    print('\n ------------------------ \n')

    # Setting early stop params
    best_val_loss = float('inf')
    patience_count = 0
    patience = 10
    print(f"Patience count starts at {patience_count}, patience is set to {patience}.")
    print('\n ------------------------ \n')

    # Setting mixed precision
    use_amp = torch.cuda.is_available() and str(device).startswith('cuda')
    scaler = GradScaler("cuda") if use_amp else None   # indica il device
    autocast_ctx = partial(autocast, "cuda") if use_amp else nullcontext
    print(f"Mixed precision enabled: {use_amp}")
    print('\n ------------------------ \n')

    # Start training
    print('--- Start training loop ---')
    for epoch in tqdm(range(start_epoch, epochs)):
        model.train()

        epoch_train_loss = epoch_router_loss = epoch_total_loss = 0
        batch_count = 0

        # Reset accuracy metrics
        accuracy_metrics['top1_train'].reset()
        accuracy_metrics['top5_train'].reset()

        accuracy_metrics['top1_val'].reset()
        accuracy_metrics['top5_val'].reset()

        if epoch == 100:
            model.router.set_keys_trainable(True)
        
        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc='Training batches')):
            if epoch < start_epoch and batch_idx < start_train_batch:
                continue

            # Extract data and labels from batch
            data, labels = data.to(device), labels.to(device)

            # transform data with data augmentation
            data = augmentation(data)

            # Forward pass
            logits = model(data)
            classification_loss = criterion(logits, labels)
            
            # Calculate variance_loss and total loss
            router_loss = calc_router_loss(model)
            model.router.disable_metrics_cache()
            total_loss = classification_loss + router_loss

            # Backward pass
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(total_loss).backward()
                # unscale before clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

            # # Track losses
            batch_class_loss = classification_loss.item()
            batch_router_loss = router_loss.item()
            batch_total_loss = total_loss.item()

            epoch_train_loss += batch_class_loss
            epoch_router_loss += batch_router_loss
            epoch_total_loss += batch_total_loss
            train_loss_history.append({
                'epoch' : epoch,
                'batch' : batch_idx,
                'classification_loss' : batch_class_loss,
                'router_loss' : batch_router_loss,
                'total_loss' : batch_total_loss
            })
            batch_count += 1

            # Update torchmetrics for training (Top-1 and Top-5)
            accuracy_metrics['top1_train'].update(logits, labels)
            if num_classes >= 5:
                accuracy_metrics['top5_train'].update(logits, labels)
            
            # Update actual batch idx
            actual_batch_idx = batch_idx

            # Save checkpoint
            if actual_batch_idx % save_checkpoint_every == 0:
                save_checkpoint(train_checkpoints_path, model, optimizer, lr_scheduler, epoch, actual_batch_idx,
                                train_loss_history, val_loss_history)

            if actual_batch_idx % log_prediction_every == 0:
                with torch.no_grad():

                    # Limit sample size
                    log_size = min(10, data.shape[0])

                    # transfer to CPU 
                    sample_data = data[:log_size].detach().cpu()
                    sample_labels = labels[:log_size].detach().cpu()
                    sample_logits = logits[:log_size].detach().cpu()

                    pred_probs = torch.softmax(sample_logits, dim=-1)

                    log_prediction_to_wandb(
                        data_batch=sample_data,
                        true_labels=sample_labels,
                        pred_labels=pred_probs,
                        batch_class_loss = batch_class_loss,
                        batch_router_loss = batch_router_loss,
                        batch_total_loss = batch_total_loss,
                        class_names=class_names,
                        num_images_to_log=log_size,
                        epoch= epoch,
                        phase = 'train'
                    )

            del data, classification_loss, router_loss, total_loss, logits

        # Calculate epoch metrics
        avg_train_class_loss = epoch_train_loss / batch_count if batch_count > 0 else 0
        avg_train_router_loss = epoch_router_loss / batch_count if batch_count > 0 else 0
        avg_train_total_loss = epoch_total_loss / batch_count if batch_count > 0 else 0
        
        # Validation phase
        model.eval()
        val_class_loss = 0.0
        val_router_loss = 0.0
        val_total_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(tqdm(val_loader, desc='Validation batches')):
                data, labels = data.to(device), labels.to(device)
                
                with autocast_ctx():
                    logits = model(data)
                
                model.router.disable_metrics_cache()
                classification_loss = criterion(logits, labels)

                router_loss = calc_router_loss(model)
                total_loss = classification_loss + router_loss

                val_class_loss += classification_loss.item()
                val_router_loss += router_loss.item()
                val_total_loss += total_loss.item()
                
                val_batches += 1

                accuracy_metrics['top1_val'].update(logits, labels)
                if num_classes >= 5:
                    accuracy_metrics['top5_val'].update(logits, labels)

                if batch_idx % log_prediction_every == 0:
                   with torch.no_grad():

                    # Limit sample size
                    log_size = min(10, data.shape[0])

                    # transfer to CPU 
                    sample_data = data[:log_size].detach().cpu()
                    sample_labels = labels[:log_size].detach().cpu()
                    sample_logits = logits[:log_size].detach().cpu()

                    pred_probs = torch.softmax(sample_logits, dim=-1)

                    log_prediction_to_wandb(
                        data_batch=sample_data,
                        true_labels=sample_labels,
                        pred_labels=pred_probs,
                        batch_class_loss = val_class_loss,
                        batch_router_loss = val_router_loss,
                        batch_total_loss = val_total_loss,
                        class_names=class_names,
                        num_images_to_log=log_size,
                        epoch= epoch,
                        phase = 'validation'
                    )

                del data, labels, logits, classification_loss, router_loss, total_loss

        # Calculate epoch metrics
        avg_val_classification_loss = val_class_loss / val_batches if val_batches > 0 else 0
        avg_val_router_loss = val_router_loss / val_batches if val_batches > 0 else 0
        avg_val_total_loss = val_total_loss / val_batches if val_batches > 0 else 0

        val_loss_history.append(avg_val_total_loss)

        # Compute final accuracy metrics
        train_top1_acc = accuracy_metrics['top1_train'].compute().item() * 100
        train_top5_acc = accuracy_metrics['top5_train'].compute().item() * 100 if num_classes >= 5 else 0

        val_top1_acc = accuracy_metrics['top1_val'].compute().item() * 100
        val_top5_acc = accuracy_metrics['top5_val'].compute().item() * 100 if num_classes >= 5 else 0
        
        # Calc router metrics every 5 epochs
        router_metrics = None
        with torch.no_grad():
            # Get sample per router metrics
            sample_data, _ = next(iter(val_loader))
            sample_data = sample_data[:8].to(device)
            _ = model(sample_data)

            _, router_metrics = calc_router_loss(model, return_stats=True)

            del sample_data
        # Early stopping check
        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            patience_count = 0

            # Save best model
            torch.save(model.state_dict(), f'{train_checkpoints_path}/best_model.pth')
        else:
            patience_count += 1

        if patience_count > patience:
            print(f'Ealry Stop')
            break

        lr = optimizer.param_groups[0]['lr']
        gradient_norm = calculate_gradient_norm(model)

        log_train_metrics_to_wandb(
            avg_train_class_loss=avg_train_class_loss,
            avg_train_router_loss=avg_train_router_loss,
            avg_train_total_loss=avg_train_total_loss,
            train_top1_acc=train_top1_acc,
            train_top5_acc = train_top5_acc,
            avg_val_classification_loss=avg_val_classification_loss,
            avg_val_router_loss=avg_val_router_loss,
            avg_val_total_loss=avg_val_total_loss,
            val_top1_acc=val_top1_acc,
            val_top5_acc = val_top5_acc,
            lr=lr,
            epoch=epoch,
            gradient_norm=gradient_norm,
            best_val_loss=best_val_loss,
            router_metrics = router_metrics
        )
        
        # Save checkpoint at end of epoch
        save_checkpoint(train_checkpoints_path, model, optimizer, lr_scheduler,
                       epoch + 1, 0, train_loss_history, val_loss_history)
        
        # Reset batch counter for next epoch
        start_train_batch = 0
        torch.cuda.empty_cache()

        lr_scheduler.step()

    
    print('Training completed!')
    return model, train_loss_history, val_loss_history

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
    epochs = 250
    pre_train_epochs = 150
    fine_tune_epochs = 100
    phase_multipliers = [1.0, 0.3]
    batch_size = 32

    print("\n--- Hyperparameters ---")
    print(f"Model: experts={num_exp}, k={kernel_size}, out_exp={out_channel_exp}, out_rout={out_channel_rout}, layers={layer_number}, patch={patch_size}, lr={lr}, dropout={dropout}, wd={weight_decay}")
    print(f"Router: ema={ema_alpha}, temp={router_temperature}, thresh={threshold}, hard={hard_threshold_router}")
    print(f"Training: epochs={epochs} (pre={pre_train_epochs}, fine={fine_tune_epochs}), phases={phase_multipliers}, batch={batch_size}\n")
    print('\n ------------------------ \n')

    train_config={
        'epochs': epochs,
        'pre_train_epochs': pre_train_epochs,
        'fine_tune_epochs': fine_tune_epochs,
        'epochs' : epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
    }

    # Initialize patch extractor
    patch_extractor = PatchExtractor(patch_size=patch_size)


    # # # Create DataLoader for training and validation of all datasets
    # # Load Cifar-10 Sets
    # cifar10_sets = get_cifar10_sets(batch_size)
    # train_datasets.append(cifar10_sets)

    # # Load TinyImageNet sets
    # tinyimagenet_sets = get_tinyimagenet_sets(batch_size)
    # train_datasets.append(tinyimagenet_sets)

    # Load pascalvoc sets
    # pascalvoc_sets = get_pascalvoc_sets()
    # train_datasets.append(pascalvoc_sets)

     # idx of the dataset : 
        # 0 -> Cifar10
        # 1 -> Tiny-ImageNet
    # dataset_idx = 0
        
    # # Define dataset and dataloader
    # train_dataset = train_datasets[dataset_idx]['datasets']['train']
    
    # train_loader = train_datasets[dataset_idx]['dataloader']['train']
    # validation_loader = train_datasets[dataset_idx]['dataloader']['val']
    # num_classes = train_datasets[dataset_idx]['num_classes']
    # class_names = train_datasets[dataset_idx]['unique_lables']

    print(f'--- Loading fake dataset ---')
    # Create fake dataset
    num_samples_train = 100
    num_samples_val = 20
    num_classes = 10
    class_names = [chr(ord('a') + i) for i in range(10)]

    # Create fake data
    fake_train_images = torch.rand(num_samples_train, 3, 32, 32)  # immagini
    fake_train_labels = torch.randint(0, num_classes, (num_samples_train,))  # labels

    fake_val_images = torch.rand(num_samples_val, 3, 32, 32)
    fake_val_labels = torch.randint(0, num_classes, (num_samples_val,))

    from torch.utils.data import TensorDataset

    fake_train_dataset = TensorDataset(fake_train_images, fake_train_labels)
    fake_val_dataset = TensorDataset(fake_val_images, fake_val_labels)

    # create DataLoader
    fake_train_loader = DataLoader(fake_train_dataset, batch_size=32, shuffle=True)
    fake_val_loader = DataLoader(fake_val_dataset, batch_size=32, shuffle=False)

    print(f'--- Dataset loaded --- \n')

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
        router_temperature = router_temperature,
        current_dataset = 'Fake dataset'
    )


    print('--- Loading router and model ---')
    # Divides current dataset in patch for initialize keys and setting input channel for model
    dataset_patch, _, _ = patch_extractor(fake_train_dataset.tensors[0])  # Extract patches 
    _, _, C, _, _ = dataset_patch.shape
    
    router = Router(num_experts=num_exp, temperature=router_temperature)
    router.initialize_keys(dataset_patch) #Initialize router keys with SSP 

    # Initialize model, optimizer (Adam) and scheduler
    # Loss initialize in training funcion (CrossEntropy)
    model = PCENetwork(
        inpt_channel= C,
        num_experts = num_exp,
        kernel_sz_exps = kernel_size,
        output_cha_exps = out_channel_exp,
        layer_number = layer_number,
        patch_size = patch_size,
        router=router,
        dropout=0.1,
        num_classes=num_classes,
        threshold=threshold,
        enable_router_metrics=True,
        hard_threshold_router = hard_threshold_router,
    )
    print('-- Router and model initialized -- \n')

    print('--- Loading optimizer, scheduler, laugmentation class and metrics for accuracy ---')
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = PCEScheduler(
        optimizer = optimizer, 
        phase_epochs=[pre_train_epochs, fine_tune_epochs],
        base_lr=lr,
        phase_multipliers=phase_multipliers)
    
    augmentation = DataAgumentation()

    metrics = setup_torchmetrics_accuracy(num_classes=num_classes, device=device)
    print('--- All loaded --- \n')

    print('--- Start with training ---')
    training(
        model=model,
        train_loader=fake_val_loader,
        val_loader=fake_val_loader,
        device=device,
        epochs=epochs,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        augmentation = augmentation,
        accuracy_metrics=metrics, 
        class_names = class_names,
        num_classes=num_classes, 
    )