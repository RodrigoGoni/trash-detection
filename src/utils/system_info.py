"""
System information utilities for reproducibility tracking.
Provides functions to gather hardware, software, and environment information.
"""

import platform
import psutil
import torch


def get_system_info() -> dict:
    """
    Get comprehensive system information for reproducibility.
    
    Returns:
        Dictionary with system, CPU, memory, disk, and GPU information
    """
    info = {
        # Python & Libraries
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        
        # System
        'platform': platform.platform(),
        'hostname': platform.node(),
        'processor': platform.processor(),
        
        # CPU
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        
        # Memory
        'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'ram_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        
        # Disk
        'disk_total_gb': round(psutil.disk_usage('/').total / (1024**3), 2),
        'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2),
        
        # CUDA
        'cuda_available': torch.cuda.is_available(),
    }
    
    # Add GPU information if CUDA is available
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        
        # GPU Memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        info['gpu_memory_gb'] = round(gpu_mem / (1024**3), 2)
    
    return info


def log_gpu_metrics_to_mlflow(step: int):
    """
    Log GPU metrics to MLflow if CUDA is available.
    
    Args:
        step: Current training step/epoch for metric tracking
    """
    if not torch.cuda.is_available():
        return
    
    try:
        import mlflow
        
        # GPU memory metrics for each GPU
        for i in range(torch.cuda.device_count()):
            # Memory allocated by PyTorch (in GB)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
            
            mlflow.log_metric(f"gpu_{i}_memory_allocated_gb", allocated, step=step)
            mlflow.log_metric(f"gpu_{i}_memory_reserved_gb", reserved, step=step)
            mlflow.log_metric(f"gpu_{i}_memory_max_allocated_gb", max_allocated, step=step)
            
            # GPU utilization (if available)
            try:
                utilization = torch.cuda.utilization(i)
                mlflow.log_metric(f"gpu_{i}_utilization_percent", utilization, step=step)
            except:
                pass  # Not all CUDA versions support this
    
    except Exception:
        # Don't fail training if GPU metrics logging fails
        pass
