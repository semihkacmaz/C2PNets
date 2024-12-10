import yaml
import torch
from pathlib import Path

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_device():
    """Get the available device (CPU/GPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_directories(config):
    """Ensure required directories exist."""
    for path_name in ["models", "images", "eos_tables"]:
        Path(config["paths"][path_name]).mkdir(parents=True, exist_ok=True)

def inverse_standard_scaler(standardized_tensor, mean, std):
    """Inverse transform standardized data."""
    return (standardized_tensor * std) + mean
