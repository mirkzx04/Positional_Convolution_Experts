import wandb as wb
import os
import io
import tarfile
import urllib.request
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import ToPILImage

import torchmetrics
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

def router_variance_loss(model, variance_weight = 0.01, return_stats = True):
    """
    Calculate the variance loss for the router weights.
    
    Args:
        expert_weights (torch.Tensor): The weights of the experts.
        variance_weight (float): Weight for the variance loss.
        
    Returns:
        torch.Tensor: The variance loss.
    """
    # Get experts weights
    metrics = model.router.get_cached_metrics()
    expert_weights = metrics['weights_filtred']

    # Calculate variance of weights experts
    variance_per_expert = torch.var(expert_weights, dim = 0)
    total_variance = torch.sum(variance_per_expert)

    if return_stats:
        wb.log({
                'router_loss/variance_per_expert_mean': variance_per_expert.mean().item(),
                'router_loss/total_variance': total_variance.item(),
                'router_loss/variance_std': variance_per_expert.std().item(),
                'router_loss/max_variance': variance_per_expert.max().item(),
                'router_loss/min_variance': variance_per_expert.min().item(),
                
                # Aggiungi metriche aggiuntive dalla cache
                'router_loss/expert_utilization_max': metrics['max_expert_utilization'],
                'router_loss/expert_utilization_min': metrics['min_expert_utilization'],
                'router_loss/routing_entropy': metrics['mean_routing_entropy'],
                'router_loss/assignment_confidence': metrics['assignment_confidence'].mean().item(),
            })
    
    return -total_variance * variance_weight

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

def log_prediction_to_wandb(data_batch, true_labels, batch_loss,
                            pred_labels, class_names = None, 
                            num_images_to_log = 10, epoch = None, phase = 'train'):
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
    if isinstance(data_batch, torch.tensor):
        data_batch = data_batch.detach().cpu()
    if isinstance(true_labels, torch.tensor):
        true_labels = true_labels.detach().cpu()
    if isinstance(pred_labels, torch.tensor):
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
        caption = f'{is_correct} True : {true_class} | Pred : {pred_class} \n batch loss : {batch_loss}'

        # Create wandb image object
        wandb_img = wb.Image(
            img_pil,
            caption=caption
        )

        wandb_images.append(wandb_img)
    
    # Log to wandb
    log_dict = {f'{phase}_predictions' : wandb_images}
    wb.log(log_dict, step = epoch)

def log_train_metrics_to_wandb(train_loss, train_accuracy, val_loss, val_accuracy, lr, epoch, gradient_norm, best_val_loss):
    """
    Log train metrics to wandb

    Args:
        train_loss (float) : training loss
        train_accuracy (float) : Accuracy of model prediction in training loop
        val_loss (float) : validation loss
        val_accuracy (float) : Accuracy of model prediction in validation loop
        lr (float) : current_learning rate
        epoch (int) : current training epoch
        best_val_loss (float) : Best validation loss
        gradient_norm (int) : Norm of gradient 
    """

    log_dict = {}
    log_dict['epoch'] = epoch

    # Add losses to logger
    log_dict['train_loss'] = train_loss
    log_dict['train_accuracy'] = train_accuracy

    log_dict['val_loss'] = val_loss
    log_dict['val_accuracy'] = val_accuracy

    log_dict['best_val_loss'] = best_val_loss

    # Add learning rate to logger
    log_dict['learning_rate'] = lr

    # Add gradient norm to logger
    log_dict['gradient_norm'] = gradient_norm
    
    # Add learning rate to logger
    log_dict['lr'] = lr

    wb.log(log_dict, step=epoch)

def get_router_metrics(model, data_batch, layer_idx):
    """
    Get metrics of the router for specific layer
    Args:
        model: PCE Model
        data_batch: Batch of img (B, C, H, W)
        layer_idx: index layer to analyze
        
    Returns:
        dict: Router metrics
    """
    model.eval()
    
    # Get metris if cache is avaible
    metrics = model.router.get_cached_metrics()

    if metrics is not None:
        return metrics
    
    # Else execute forward pass
    if data_batch is not None:
        with torch.no_grad():
            model.router.enable_metrics_cache()
            _ = model(data_batch)
            return model.router.get_cached_metrics()
    
    return None
        
def log_router_to_wandb(metrics, layer_idx, epoch, log_individual_experts=True):
    """
    Log metriche del router su wandb
    
    Args:
        metrics: Dizionario di metriche dal router
        layer_idx: Indice del layer
        epoch: Epoca corrente
        log_individual_experts: Se loggare utilizzo individuale degli esperti
    """
    if metrics is None:
        return
    
    # Principale Metrics
    log_dict = {
        # Experts
        f'router/L{layer_idx}/expert_utilization_max': metrics['max_expert_utilization'] * 100,
        f'router/L{layer_idx}/expert_utilization_min': metrics['min_expert_utilization'] * 100,
        f'router/L{layer_idx}/expert_utilization_std': metrics['utilization_std'] * 100,
        f'router/L{layer_idx}/expert_utilization_entropy': metrics['utilization_entropy'],
        
        # Quality routing
        f'router/L{layer_idx}/routing_entropy': metrics['mean_routing_entropy'],
        f'router/L{layer_idx}/assignment_confidence': metrics['assignment_confidence'].mean().item(),
        f'router/L{layer_idx}/low_confidence_patches_pct': metrics['low_confidence_patches_pct'],
        
        # Sparsity and activation
        f'router/L{layer_idx}/sparsity_pct': metrics['sparsity_level'] * 100,
        f'router/L{layer_idx}/active_experts_per_patch': metrics['active_experts_per_patch_mean'],
        
        # Similarities of keys
        f'router/L{layer_idx}/keys_max_similarity': metrics['keys_max_similarity'],
        f'router/L{layer_idx}/keys_mean_similarity': metrics['keys_mean_similarity'],
        
        # Router params
        f'router/L{layer_idx}/threshold': metrics['threshold'],
        f'router/L{layer_idx}/max_weight_filtered': metrics['mean_max_weight_filtered']
    }
    
    if log_individual_experts:
        for expert_idx in range(len(metrics['expert_utilization'])):
            log_dict[f'router/L{layer_idx}/expert_{expert_idx}_utilization'] = \
                metrics['expert_utilization'][expert_idx].item() * 100
    
    wb.log(log_dict, step=epoch)

def check_router_health(metrics, layer_idx):
    """
    Verify health of router

    Args:
        metrics: Metriche del router
        layer_idx: Indice del layer
        
    Returns:
        tuple: (health_score, issues_list, warnings_list)
    """

    if metrics is None:
        return 0.0, ['METRICS_ERROR'], []
    
    issues = []
    warnings = []
    health_score = 1.0
    
    # 1. Expert Dominance (un esperto domina troppo)
    max_util = metrics['max_expert_utilization']
    min_util = metrics['min_expert_utilization']
    
    if max_util > 0.8:
        issues.append('EXPERT_DOMINANCE')
        health_score -= 0.3
    elif max_util > 0.6:
        warnings.append('HIGH_EXPERT_UTILIZATION')
        health_score -= 0.1
    
    # 2. Expert Underutilization (alcuni esperti inutilizzati)
    if min_util < 0.02:
        issues.append('EXPERT_UNDERUTILIZATION')
        health_score -= 0.2
    elif min_util < 0.05:
        warnings.append('LOW_EXPERT_UTILIZATION')
        health_score -= 0.1
    
    # 3. Low Routing Entropy (routing troppo deterministico)
    entropy = metrics['mean_routing_entropy']
    if entropy < 0.5:
        issues.append('LOW_ROUTING_ENTROPY')
        health_score -= 0.3
    elif entropy < 1.0:
        warnings.append('MODERATE_ROUTING_ENTROPY')
        health_score -= 0.1
    
    # 4. Key Similarity (esperti troppo simili)
    max_sim = metrics['keys_max_similarity']
    if max_sim > 0.95:
        issues.append('HIGH_KEY_SIMILARITY')
        health_score -= 0.3
    elif max_sim > 0.85:
        warnings.append('MODERATE_KEY_SIMILARITY')
        health_score -= 0.1
    
    # 5. Low Assignment Confidence
    confidence = metrics['assignment_confidence'].mean().item()
    if confidence < 0.3:
        issues.append('LOW_ASSIGNMENT_CONFIDENCE')
        health_score -= 0.2
    elif confidence < 0.5:
        warnings.append('MODERATE_ASSIGNMENT_CONFIDENCE')
        health_score -= 0.1
    
    # 6. High Sparsity (threshold troppo alto)
    sparsity = metrics['sparsity_level']
    if sparsity > 0.9:
        issues.append('EXCESSIVE_SPARSITY')
        health_score -= 0.3
    elif sparsity > 0.7:
        warnings.append('HIGH_SPARSITY')
        health_score -= 0.1
    
    # Assicura che health_score sia tra 0 e 1
    health_score = max(0.0, min(1.0, health_score))
    
    return health_score, issues, warnings

def quick_router_check(model, data_batch, layer_idx, epoch):
    """
    Quick check of router with complete logging

    Args:
        model : PCE Model
        data_batch : Batch of image
        layer_idx : layer to analyze
        epoch : current epoch
    """

    metrics = get_router_metrics(model, data_batch, layer_idx)

    if metrics is None:
        return 
    
    # Check router health
    health_score, issues, warnings = check_router_health(metrics, layer_idx)

    # Log on wandb (only principale metrics)
    wb.log({
        f'router/L{layer_idx}/quick_health_score': health_score,
        f'router/L{layer_idx}/quick_entropy': metrics['mean_routing_entropy'],
        f'router/L{layer_idx}/quick_confidence': metrics['assignment_confidence'].mean().item(),
        f'router/L{layer_idx}/quick_sparsity': metrics['sparsity_level'],
        f'router/L{layer_idx}/issues_count': len(issues),
        f'router/L{layer_idx}/warnings_count': len(warnings),
        f'router/L{layer_idx}/overall_status': 1 if len(issues) == 0 else 0
    }, step=epoch)

def detailed_router_analysis(model, data_batch, layer_idx, epoch):
    """
    Detailed analysis of router with complete logging

    Args:
        model : PCE model
        data_batch : batch of image
        layer_idx : Layaer to analyze
        epoch : Current epoch
    """

    metrics = get_router_metrics(model, data_batch, layer_idx)

    if metrics is None:
        return
    
    log_router_to_wandb(metrics, layer_idx, epoch)

    health_score, issues, warnings = check_router_health(metrics, layer_idx)

    wb.log({
        f'router/L{layer_idx}/quick_health_score': health_score,
        f'router/L{layer_idx}/quick_entropy': metrics['mean_routing_entropy'],
        f'router/L{layer_idx}/quick_confidence': metrics['assignment_confidence'].mean().item(),
        f'router/L{layer_idx}/quick_sparsity': metrics['sparsity_level'],
        f'router/L{layer_idx}/issues_count': len(issues),
        f'router/L{layer_idx}/warnings_count': len(warnings),
        f'router/L{layer_idx}/overall_status': 1 if len(issues) == 0 else 0
    }, step=epoch)

def training(
        model, 
        train_loader, 
        val_loader, 
        device,
        epochs,
        optimizer, 
        lr_scheduler,
        augmentation,
        metrics,
        class_names,
        train_checkpoints_path = './checkpoints'):
    
    # Setting train and val loader
    train_loader = train_loader
    val_loader = val_loader 

    # enables router cache
    model.router.enable_metrics_cache()

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

    # Setting how often to do training, model and router logging
    save_checkpoint_every = 5
    log_prediction_every = 100
    log_metrics_every = 50

    ROUTER_QUICK_CHECK_EVERY = 15      
    ROUTER_DETAILED_ANALYSIS_EVERY = 30 
    ROUTER_SAMPLE_SIZE = 6

    # Setting early stop params
    best_val_loss = float('inf')
    patience_count = 0
    patience = 10
 
    # Start training
    for epoch in tqdm(range(start_epoch, epochs)):
        model.train()

        train_loss = val_loss = 0
        epoch_train_loss = 0
        batch_count = 0

        train_correct = 0
        train_total = 0

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
            loss = criterion(logits, labels)
            
            # Calculate variance_loss and total loss
            variance_loss = router_variance_loss(model)
            batch_loss = variance_loss + loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()
            lr_scheduler.step()

            # # Track losses
            epoch_train_loss += batch_loss
            train_loss_history.append(batch_loss)
            batch_count += 1

            # Prepare data to calculate accuracy
            if 'train_correct' in locals():
                _, predicted = torch.max(logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Update actual batch idx
            if epoch == start_epoch:
                actual_batch_idx = current_batch_start + batch_idx
            else:
                actual_batch_idx = batch_idx

            # Save checkpoint
            if actual_batch_idx % save_checkpoint_every == 0:
                save_checkpoint(train_checkpoints_path, model, optimizer, lr_scheduler, epoch, actual_batch_idx,
                                train_loss_history, val_loss_history)

            # Log intermediate metrics
            if actual_batch_idx % log_metrics_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                wb.log({
                    'batch_loss' : batch_loss,
                    'learning_rate' : current_lr,
                    'epoch' : epoch,
                    'current_batch' : actual_batch_idx
                })

            if actual_batch_idx % log_prediction_every == 0:
                with torch.no_grad():
                    pred_probs = torch.softmax(logits[:15], dim=-1)
                    log_prediction_to_wandb(
                        data_batch=data[:15],
                        true_labels=labels[:15],
                        batch_loss=batch_loss,
                        pred_labels=pred_probs[:15],
                        class_names=class_names,
                        num_images_to_log=10,
                        epoch= epoch,
                        phase = 'train'
                    )

        # Calculate epoch metrics
        avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else 0
        train_accuracy = (train_correct / train_total * 100) if 'train_correct' in locals() else 0
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (data, labels) in tqdm(val_loader, desc='Validation batches'):
                data, labels = data.to(device), labels.to(device)
                
                logits = model(data)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                val_batches += 1

                # Prepare data to calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                if batch_idx % log_prediction_every == 0:
                    with torch.no_grad():
                        pred_probs = torch.softmax(logits[:15], dim=-1)
                        log_prediction_to_wandb(
                            data_batch=data[:15],
                            true_labels=labels[:15],
                            batch_loss=val_loss,
                            pred_labels=pred_probs[:15],
                            class_name = None,
                            num_images_to_log=10,
                            epoch= epoch,
                            phase = 'validation'
                        )
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        val_accuracy = (val_correct / val_total * 100) if val_total > 0 else 0
        val_loss_history.append(avg_val_loss)
                
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_count = 0

            # Save best model
            torch.save(model.state_dict(), f'{train_checkpoints_path}/best_model.pth')
        else:
            patience_count += 1

        # Router quick check every 15 epochs
        if epoch % ROUTER_QUICK_CHECK_EVERY == 0 and epoch > 0:
            sample_data, _ = next(iter(val_loader))
            sample_data = sample_data[:ROUTER_SAMPLE_SIZE].to(device)

            model.router.enable_metrics_cache()

            with torch.no_grad():
                # Quick check only first layer
                quick_router_check(
                    model, sample_data, layer_idx = 0, epoch = epoch
                )

        # Router detailed analysis every 30 epochs
        if epoch % ROUTER_DETAILED_ANALYSIS_EVERY == 0 and epoch > 0:
            sample_data, _ = next(iter(val_loader))
            sample_data = sample_data[:ROUTER_SAMPLE_SIZE].to(device)

            model.router.enable_metrics_cache()

            with torch.no_grad():
                # Detailed check only first layer
                detailed_router_analysis(
                    model, sample_data, layer_idx = 0, epoch = epoch
                )

        lr = optimizer.param_groups[0]['lr']
        gradient_norm = calculate_gradient_norm(model)

        log_train_metrics_to_wandb(
            train_loss=avg_train_loss,
            train_accuracy=train_accuracy,
            val_loss=avg_val_loss,
            val_accuracy=val_accuracy,
            lr=lr,
            epoch=epoch,
            best_val_loss=best_val_loss,
            gradient_norm=gradient_norm,
        )
        
        # Save checkpoint at end of epoch
        save_checkpoint(train_checkpoints_path, model, optimizer, lr_scheduler,
                       epoch + 1, 0, train_loss_history, val_loss_history)
        
        # Reset batch counter for next epoch
        start_train_batch = 0
    
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
    cifar10_sets = get_cifar10_sets(batch_size)
    train_datasets.append(cifar10_sets)

    # Load TinyImageNet sets
    tinyimagenet_sets = get_tinyimagenet_sets(batch_size)
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
    num_classes = train_datasets[dataset_idx]['num_classes']
    class_names = train_datasets[dataset_idx]['unique_lables']

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

    # Divides current dataset in patch for initialize keys and setting input channel for model
    dataset_patch, _, _ = patch_esxtractor(train_dataset.data)  # Extract patches 
    _, _, C, _, _ = dataset_patch.shape
    
    router = Router(num_experts=num_exp, out_channel_key=out_channel_exp)
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

    metrics = setup_torchmetrics_accuracy(num_classes=num_classes, device= 'cuda' if torch.cuda.is_available() else 'cpu')

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
        class_names = class_names,
        metrics = metrics
    )