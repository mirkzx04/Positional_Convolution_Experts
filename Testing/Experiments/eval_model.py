import sys
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet18

# Setup paths cleanly using pathlib
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Model.PCE import PCENetwork
from Testing.Experiments.InferenceDataset import InferenceDataset

NUM_EXP = [4, 8, 16]
INFERENCE_SET_PTH = Path('./Testing/inference_samples')
CHECKPOINT_DIR = BASE_DIR.parent / 'Saved_checkpoint'

DEVICE = th.device('cuda' if th.cuda.is_available() else 'cpu')
print(f'++++ RUNNING ON {DEVICE} ++++')

inference_transformation = T.Compose([
    T.Resize(256, interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = InferenceDataset(
    str(INFERENCE_SET_PTH / 'images'), 
    str(INFERENCE_SET_PTH / 'annotations.txt'),
    mapping_json_path='./class_mapping.json',
    transform=inference_transformation
)
test_loader = DataLoader(dataset, batch_size=1)


def load_state_dict_cleaned(checkpoint_pth):
    """Load checkpoint and remove 'model.' prefix from keys."""
    checkpoint = th.load(checkpoint_pth, map_location=DEVICE)
    state_dict = checkpoint.get('state_dict', checkpoint)

    return {k.replace('model.', ''): v for k, v in state_dict.items()}


def get_moe_model(exp):
    """Instantiate a MoE model based on the number of experts."""
    return PCENetwork(
        num_experts=exp,
        layer_number=8,
        patch_size=16,
        num_classes=200,
        router_temp=0.85,
        capacity_factor_train=2.0,
        capacity_factor_val=2.0
    )


def evaluate_model(model, dataloader, device, warmup_steps=5):
    """Generic inference loop for latency and accuracy calculation."""
    model.eval()
    model.to(device)

    correct_predictions = 0
    total_samples = 0
    latencies = []

    with th.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if device.type == 'cuda':
                th.cuda.synchronize() 

            start_time = time.perf_counter()
            outputs = model(images)
            
            # Handle both MoE (tuple output) and Dense (tensor output)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            if device.type == 'cuda':
                th.cuda.synchronize()
                
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

            predicted = logits.argmax(dim=1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    # Drop warmup steps for accurate latency metrics
    valid_latencies = latencies[warmup_steps:] if len(latencies) > warmup_steps else latencies
    
    return {
        "Accuracy (%)": round((correct_predictions / total_samples) * 100, 2) if total_samples > 0 else 0.0,
        "Avg Latency (ms)": round(np.mean(valid_latencies), 2),
        "Std Latency (ms)": round(np.std(valid_latencies), 2)
    }


def compute_params():
    """Count total and trainable parameters for MoE models."""
    params_out = {}

    for exp in NUM_EXP:
        model = get_moe_model(exp)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_out[exp] = {'total_params': total_params, 'trainable_params': trainable_params}

    return params_out


def inference_moe():
    """Run inference for all MoE expert configurations."""
    results_out = {}

    for exp in NUM_EXP:
        checkpoint_pth = CHECKPOINT_DIR / f'checkpoints-{exp}EXP' / 'last.ckpt'
        
        model = get_moe_model(exp)
        model.load_state_dict(load_state_dict_cleaned(checkpoint_pth))
        
        print(f"Evaluating MoE model with {exp} experts...")
        results_out[exp] = evaluate_model(model, test_loader, DEVICE)

    return results_out


def inference_dense(): 
    """Run inference for the baseline Dense model (ResNet18)."""
    checkpoint_pth = CHECKPOINT_DIR / 'checkpointsResNet18' / 'last.ckpt'
    
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 200)
    model.load_state_dict(load_state_dict_cleaned(checkpoint_pth))

    print("Evaluating Dense model (ResNet18)...")
    return {'dense': evaluate_model(model, test_loader, DEVICE)}
