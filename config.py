"""
Centralized hyperparameter configuration for PCE project.
All training hyperparameters are defined here for easy modification.
"""
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
import os


@dataclass
class ModelConfig:
    """Model architecture hyperparameters"""
    num_experts: int = 8
    layer_number: int = 4
    patch_size: int = 16
    dropout: float = 0.1
    num_classes: int = 10
    hidden_size: int = 128
    
    # Router parameters
    router_temp: float = 1.0
    load_factor: float = 0.01
    noise_epsilon: float = 1e-2
    noise_std: float = 1.0
    capacity_factor_train: float = 1.25
    capacity_factor_val: float = 2.0


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Basic training
    lr: float = 3e-4
    weight_decay: float = 1.5e-4  # 0.00015
    train_epochs: int = 200
    batch_size: int = 128
    accumulate_grad_batches: int = 1
    
    # Scheduler parameters
    scheduler_type: str = 'cosine'  # 'cosine', 'multistep', 'polynomial'
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01
    
    # Loss scheduling
    uniform_epoch: int = 30
    end_loss_epoch: int = 100
    base_loss_weight: float = 0.0
    tgt_loss_weight: float = 1.0
    base_tau: float = 1.0
    tgt_tau: float = 0.1
    end_tau_epoch: int = 100


@dataclass
class LossConfig:
    """Loss function hyperparameters"""
    loss_type: str = 'soft_ce'  # 'soft_ce', 'focal', 'kl_div'
    label_smoothing: float = 0.02  # Very low as requested
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0


@dataclass
class AugmentationConfig:
    """Data augmentation hyperparameters"""
    use_augmentation: bool = True
    
    # RandAugment
    use_randaugment: bool = True
    randaugment_n: int = 2
    randaugment_m: int = 9
    
    # Random Erasing
    use_random_erasing: bool = True
    erasing_prob: float = 0.25
    erasing_scale: tuple = (0.02, 0.33)
    erasing_ratio: tuple = (0.3, 3.3)
    
    # MixUp & CutMix
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.5
    
    # Basic augmentations (fallback)
    horizontal_flip_prob: float = 0.5
    rotation_degrees: int = 15
    color_jitter: Dict[str, float] = field(default_factory=lambda: {
        'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1
    })


@dataclass
class OptimizationConfig:
    """Optimization hyperparameters following CIFAR-10 best practices"""
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # Gradient clipping
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = 'norm'
    
    # Weight decay exclusions
    exclude_bias_from_wd: bool = True
    exclude_norm_from_wd: bool = True
    
    # Drop path (if applicable)
    drop_path_rate: float = 0.1


@dataclass
class DataConfig:
    """Dataset hyperparameters"""
    dataset_name: str = 'cifar10'  # 'cifar10', 'tiny_imagenet'
    data_dir: str = './Data'
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # CIFAR-10 specific
    cifar_normalize: bool = True
    cifar_mean: tuple = (0.4914, 0.4822, 0.4465)
    cifar_std: tuple = (0.2470, 0.2435, 0.2616)


@dataclass
class LoggingConfig:
    """Logging and monitoring hyperparameters"""
    use_wandb: bool = True
    wandb_project: str = 'PCE-CIFAR10'
    wandb_entity: Optional[str] = None
    experiment_name: Optional[str] = None
    
    log_every_n_steps: int = 50
    save_top_k: int = 3
    monitor: str = 'val_acc'
    mode: str = 'max'
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_last: bool = True
    save_weights_only: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Runtime
    seed: int = 42
    device: str = 'auto'  # 'auto', 'cuda', 'cpu', 'mps'
    precision: str = '16-mixed'  # '32', '16-mixed', 'bf16-mixed'
    
    def save_config(self, path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'loss': self.loss.__dict__,
            'augmentation': self.augmentation.__dict__,
            'optimization': self.optimization.__dict__,
            'data': self.data.__dict__,
            'logging': self.logging.__dict__,
            'seed': self.seed,
            'device': self.device,
            'precision': self.precision
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_config(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        
        # Update nested configs
        if 'model' in config_dict:
            for k, v in config_dict['model'].items():
                setattr(config.model, k, v)
        
        if 'training' in config_dict:
            for k, v in config_dict['training'].items():
                setattr(config.training, k, v)
                
        if 'loss' in config_dict:
            for k, v in config_dict['loss'].items():
                setattr(config.loss, k, v)
                
        if 'augmentation' in config_dict:
            for k, v in config_dict['augmentation'].items():
                setattr(config.augmentation, k, v)
                
        if 'optimization' in config_dict:
            for k, v in config_dict['optimization'].items():
                setattr(config.optimization, k, v)
                
        if 'data' in config_dict:
            for k, v in config_dict['data'].items():
                setattr(config.data, k, v)
                
        if 'logging' in config_dict:
            for k, v in config_dict['logging'].items():
                setattr(config.logging, k, v)
        
        # Update top-level configs
        config.seed = config_dict.get('seed', config.seed)
        config.device = config_dict.get('device', config.device)
        config.precision = config_dict.get('precision', config.precision)
        
        return config


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for command line configuration"""
    parser = argparse.ArgumentParser(description='PCE Training Configuration')
    
    # Config file
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--save_config', type=str, help='Path to save current configuration')
    
    # Model parameters
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts per layer')
    parser.add_argument('--layer_number', type=int, default=4, help='Number of PCE layers')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden dimension size')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1.5e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    
    # Loss parameters
    parser.add_argument('--label_smoothing', type=float, default=0.02, help='Label smoothing factor')
    parser.add_argument('--loss_type', type=str, default='soft_ce', choices=['soft_ce', 'focal', 'kl_div'])
    
    # Augmentation parameters
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='MixUp alpha')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='CutMix alpha')
    parser.add_argument('--randaugment_n', type=int, default=2, help='RandAugment N')
    parser.add_argument('--randaugment_m', type=int, default=9, help='RandAugment M')
    
    # EMA parameters
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate')
    parser.add_argument('--no_ema', action='store_true', help='Disable EMA')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'tiny_imagenet'])
    parser.add_argument('--data_dir', type=str, default='./Data', help='Data directory')
    
    # Logging
    parser.add_argument('--wandb_project', type=str, default='PCE-CIFAR10', help='WandB project name')
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    
    # Runtime
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--precision', type=str, default='16-mixed', help='Training precision')
    
    return parser


def update_config_from_args(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Update configuration from command line arguments"""
    
    # Model parameters
    if args.num_experts is not None:
        config.model.num_experts = args.num_experts
    if args.layer_number is not None:
        config.model.layer_number = args.layer_number
    if args.patch_size is not None:
        config.model.patch_size = args.patch_size
    if args.dropout is not None:
        config.model.dropout = args.dropout
    if args.hidden_size is not None:
        config.model.hidden_size = args.hidden_size
    
    # Training parameters
    if args.lr is not None:
        config.training.lr = args.lr
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.epochs is not None:
        config.training.train_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    # Loss parameters
    if args.label_smoothing is not None:
        config.loss.label_smoothing = args.label_smoothing
    if args.loss_type is not None:
        config.loss.loss_type = args.loss_type
    
    # Augmentation parameters
    if args.mixup_alpha is not None:
        config.augmentation.mixup_alpha = args.mixup_alpha
    if args.cutmix_alpha is not None:
        config.augmentation.cutmix_alpha = args.cutmix_alpha
    if args.randaugment_n is not None:
        config.augmentation.randaugment_n = args.randaugment_n
    if args.randaugment_m is not None:
        config.augmentation.randaugment_m = args.randaugment_m
    
    # EMA parameters
    if args.ema_decay is not None:
        config.optimization.ema_decay = args.ema_decay
    if args.no_ema:
        config.optimization.use_ema = False
    
    # Dataset
    if args.dataset is not None:
        config.data.dataset_name = args.dataset
    if args.data_dir is not None:
        config.data.data_dir = args.data_dir
    
    # Logging
    if args.wandb_project is not None:
        config.logging.wandb_project = args.wandb_project
    if args.experiment_name is not None:
        config.logging.experiment_name = args.experiment_name
    if args.no_wandb:
        config.logging.use_wandb = False
    
    # Runtime
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.precision is not None:
        config.precision = args.precision
    
    return config


# Predefined configurations for common scenarios
def get_quick_test_config() -> ExperimentConfig:
    """Configuration for quick testing/debugging"""
    config = ExperimentConfig()
    config.training.train_epochs = 5
    config.training.batch_size = 32
    config.logging.use_wandb = False
    config.augmentation.use_mixup = False
    config.optimization.use_ema = False
    return config


def get_cifar10_best_config() -> ExperimentConfig:
    """Best configuration for CIFAR-10 following modern practices"""
    config = ExperimentConfig()
    
    # Model - optimized for CIFAR-10
    config.model.num_experts = 8
    config.model.layer_number = 4
    config.model.patch_size = 16
    config.model.dropout = 0.1
    config.model.hidden_size = 256
    
    # Training - CIFAR-10 best practices
    config.training.lr = 3e-4
    config.training.weight_decay = 5e-4
    config.training.train_epochs = 300
    config.training.batch_size = 128
    config.training.warmup_epochs = 5
    
    # Loss - aggressive label smoothing for CIFAR-10
    config.loss.label_smoothing = 0.1
    
    # Augmentation - heavy augmentation for CIFAR-10
    config.augmentation.randaugment_m = 15
    config.augmentation.mixup_alpha = 0.8
    config.augmentation.cutmix_alpha = 1.0
    
    # Optimization - all best practices enabled
    config.optimization.ema_decay = 0.9999
    config.optimization.drop_path_rate = 0.2
    
    return config


def get_tiny_imagenet_config() -> ExperimentConfig:
    """Configuration optimized for Tiny ImageNet"""
    config = ExperimentConfig()
    
    # Model - larger for Tiny ImageNet
    config.model.num_experts = 12
    config.model.layer_number = 6
    config.model.patch_size = 32
    config.model.hidden_size = 384
    config.model.num_classes = 200
    
    # Training
    config.training.lr = 1e-4
    config.training.train_epochs = 400
    config.training.batch_size = 64
    
    # Data
    config.data.dataset_name = 'tiny_imagenet'
    
    return config
