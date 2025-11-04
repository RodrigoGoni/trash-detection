"""
Optimizer and scheduler creation utilities.
Provides factory functions for creating optimizers and learning rate schedulers.
"""

import torch


def create_optimizer(model_parameters, config: dict):
    """
    Create optimizer based on configuration.
    
    Args:
        model_parameters: Model parameters to optimize
        config: Configuration dictionary with optimizer settings
        
    Returns:
        torch.optim.Optimizer instance
        
    Raises:
        ValueError: If optimizer type is not supported
    """
    optimizer_type = config['type'].lower()
    lr = config['lr']
    
    if optimizer_type == 'sgd':
        return torch.optim.SGD(
            model_parameters,
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0.0005)
        )
    
    elif optimizer_type == 'adam':
        return torch.optim.Adam(
            model_parameters,
            lr=lr,
            betas=config.get('betas', [0.9, 0.999]),
            weight_decay=config.get('weight_decay', 0.0001),
            eps=config.get('eps', 1e-8)
        )
    
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model_parameters,
            lr=lr,
            betas=config.get('betas', [0.9, 0.999]),
            weight_decay=config.get('weight_decay', 0.01),
            eps=config.get('eps', 1e-8)
        )
    
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. "
                        f"Supported types: sgd, adam, adamw")


def create_scheduler(optimizer, config: dict, num_epochs: int):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary with scheduler settings
        num_epochs: Total number of training epochs
        
    Returns:
        torch.optim.lr_scheduler instance
        
    Raises:
        ValueError: If scheduler type is not supported
    """
    scheduler_type = config['type'].lower()
    
    if scheduler_type == 'step_lr':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['gamma']
        )
    
    elif scheduler_type == 'cosine_annealing':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=config.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == 'cosine_annealing_warmup':
        warmup_epochs = config.get('warmup_epochs', 5)
        
        # Main cosine annealing scheduler
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Linear warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        # Combine warmup and main scheduler
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}. "
                        f"Supported types: step_lr, cosine_annealing, cosine_annealing_warmup")
