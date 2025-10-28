"""
Model configuration and initialization metadata.

This module provides metadata about NeuralForecast models to correctly
initialize them with appropriate parameters.
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class ModelInitConfig:
    """Configuration for model initialization.
    
    Attributes:
        name: Model name
        requires_n_series: Whether model requires n_series parameter
        supports_hist_exog: Whether model supports historical exogenous variables
        supports_futr_exog: Whether model supports future exogenous variables  
        supports_stat_exog: Whether model supports static exogenous variables
        special_init: Whether model requires special initialization (e.g., HINT)
        custom_params: Additional custom parameters needed
        notes: Additional notes about the model
    """
    name: str
    requires_n_series: bool = False
    supports_hist_exog: bool = True
    supports_futr_exog: bool = False
    supports_stat_exog: bool = False
    special_init: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# Model configurations based on NeuralForecast source code
MODEL_CONFIGS: Dict[str, ModelInitConfig] = {
    # MLP-based models
    'MLP': ModelInitConfig(
        name='MLP',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="Standard MLP model with full exogenous support"
    ),
    'NHITS': ModelInitConfig(
        name='NHITS',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="NHITS with full exogenous support"
    ),
    'NBEATSx': ModelInitConfig(
        name='NBEATSx',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="NBEATS extended with exogenous variables"
    ),
    'NBEATS': ModelInitConfig(
        name='NBEATS',
        supports_hist_exog=False,
        supports_futr_exog=False,
        supports_stat_exog=False,
        notes="Original NBEATS without exogenous support"
    ),
    
    # Multivariate models (require n_series)
    'MLPMultivariate': ModelInitConfig(
        name='MLPMultivariate',
        requires_n_series=True,
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="Multivariate MLP requires n_series parameter"
    ),
    'TimeXer': ModelInitConfig(
        name='TimeXer',
        requires_n_series=True,
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="TimeXer multivariate model requires n_series parameter"
    ),
    'TSMixerx': ModelInitConfig(
        name='TSMixerx',
        requires_n_series=True,
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="TSMixerx multivariate model requires n_series parameter"
    ),
    'TSMixer': ModelInitConfig(
        name='TSMixer',
        requires_n_series=True,
        supports_hist_exog=False,
        supports_futr_exog=False,
        supports_stat_exog=False,
        notes="TSMixer without exogenous support"
    ),
    
    # RNN-based models
    'LSTM': ModelInitConfig(
        name='LSTM',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="LSTM with full exogenous support"
    ),
    'GRU': ModelInitConfig(
        name='GRU',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="GRU with full exogenous support"
    ),
    'RNN': ModelInitConfig(
        name='RNN',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="RNN with full exogenous support"
    ),
    'DilatedRNN': ModelInitConfig(
        name='DilatedRNN',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="Dilated RNN with full exogenous support"
    ),
    
    # CNN-based models
    'TCN': ModelInitConfig(
        name='TCN',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="Temporal Convolutional Network with full exogenous support"
    ),
    'BiTCN': ModelInitConfig(
        name='BiTCN',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="Bidirectional TCN with full exogenous support"
    ),
    
    # Probabilistic models
    'DeepAR': ModelInitConfig(
        name='DeepAR',
        supports_hist_exog=False,  # EXOGENOUS_HIST = False in source
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="DeepAR does NOT support hist_exog_list (raises exception in __init__)"
    ),
    'DeepNPTS': ModelInitConfig(
        name='DeepNPTS',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="Deep Non-Parametric Time Series with full exogenous support"
    ),
    
    # Transformer-based models
    'VanillaTransformer': ModelInitConfig(
        name='VanillaTransformer',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="Vanilla Transformer with full exogenous support"
    ),
    'Informer': ModelInitConfig(
        name='Informer',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="Informer with full exogenous support"
    ),
    'Autoformer': ModelInitConfig(
        name='Autoformer',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="Autoformer with full exogenous support"
    ),
    'FEDformer': ModelInitConfig(
        name='FEDformer',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="FEDformer with full exogenous support"
    ),
    'PatchTST': ModelInitConfig(
        name='PatchTST',
        supports_hist_exog=False,
        supports_futr_exog=False,
        supports_stat_exog=False,
        notes="PatchTST without exogenous support"
    ),
    'iTransformer': ModelInitConfig(
        name='iTransformer',
        supports_hist_exog=False,
        supports_futr_exog=False,
        supports_stat_exog=False,
        notes="iTransformer without exogenous support"
    ),
    
    # Specialized models
    'TFT': ModelInitConfig(
        name='TFT',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="Temporal Fusion Transformer with full exogenous support"
    ),
    'TiDE': ModelInitConfig(
        name='TiDE',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="TiDE with full exogenous support"
    ),
    'TimesNet': ModelInitConfig(
        name='TimesNet',
        supports_hist_exog=False,  # Raises exception despite having parameter
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="TimesNet does NOT support hist_exog_list (raises exception in __init__)"
    ),
    'KAN': ModelInitConfig(
        name='KAN',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="Kolmogorov-Arnold Networks with full exogenous support"
    ),
    
    # Linear models
    'DLinear': ModelInitConfig(
        name='DLinear',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="DLinear with full exogenous support"
    ),
    'NLinear': ModelInitConfig(
        name='NLinear',
        supports_hist_exog=True,
        supports_futr_exog=True,
        supports_stat_exog=True,
        notes="NLinear with full exogenous support"
    ),
    
    # Special models
    'HINT': ModelInitConfig(
        name='HINT',
        special_init=True,
        supports_hist_exog=False,
        supports_futr_exog=False,
        supports_stat_exog=False,
        notes="HINT requires special initialization with S matrix, base model, and reconciliation strategy"
    ),
    
    # Other models
    'StemGNN': ModelInitConfig(
        name='StemGNN',
        supports_hist_exog=False,
        supports_futr_exog=False,
        supports_stat_exog=False,
        notes="StemGNN without exogenous support"
    ),
    'SOFTS': ModelInitConfig(
        name='SOFTS',
        supports_hist_exog=False,
        supports_futr_exog=False,
        supports_stat_exog=False,
        notes="SOFTS without exogenous support"
    ),
    'TimeLLM': ModelInitConfig(
        name='TimeLLM',
        supports_hist_exog=False,
        supports_futr_exog=False,
        supports_stat_exog=False,
        notes="TimeLLM without exogenous support"
    ),
    'TimeMixer': ModelInitConfig(
        name='TimeMixer',
        supports_hist_exog=False,
        supports_futr_exog=False,
        supports_stat_exog=False,
        notes="TimeMixer without exogenous support"
    ),
    'RMoK': ModelInitConfig(
        name='RMoK',
        supports_hist_exog=False,
        supports_futr_exog=False,
        supports_stat_exog=False,
        notes="RMoK without exogenous support"
    ),
}


def get_model_config(model_name: str) -> ModelInitConfig:
    """Get initialization configuration for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelInitConfig for the model
        
    Raises:
        KeyError: If model configuration not found
    """
    if model_name not in MODEL_CONFIGS:
        raise KeyError(f"Model configuration not found for: {model_name}")
    return MODEL_CONFIGS[model_name]


def get_models_requiring_n_series() -> Set[str]:
    """Get set of model names that require n_series parameter.
    
    Returns:
        Set of model names requiring n_series
    """
    return {name for name, config in MODEL_CONFIGS.items() if config.requires_n_series}


def get_models_supporting_hist_exog() -> Set[str]:
    """Get set of model names that support historical exogenous variables.
    
    Returns:
        Set of model names supporting hist_exog
    """
    return {name for name, config in MODEL_CONFIGS.items() if config.supports_hist_exog}


def get_models_with_special_init() -> Set[str]:
    """Get set of model names that require special initialization.
    
    Returns:
        Set of model names with special initialization
    """
    return {name for name, config in MODEL_CONFIGS.items() if config.special_init}


def save_model_configs_to_yaml(path: Path) -> None:
    """Save model configurations to YAML file.
    
    Args:
        path: Path to save YAML file
    """
    configs_dict = {}
    for name, config in MODEL_CONFIGS.items():
        configs_dict[name] = {
            'requires_n_series': config.requires_n_series,
            'supports_hist_exog': config.supports_hist_exog,
            'supports_futr_exog': config.supports_futr_exog,
            'supports_stat_exog': config.supports_stat_exog,
            'special_init': config.special_init,
            'custom_params': config.custom_params,
            'notes': config.notes
        }
    
    with open(path, 'w') as f:
        yaml.dump(configs_dict, f, default_flow_style=False, sort_keys=True)


def load_model_configs_from_yaml(path: Path) -> Dict[str, ModelInitConfig]:
    """Load model configurations from YAML file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Dictionary of model configurations
    """
    with open(path, 'r') as f:
        configs_dict = yaml.safe_load(f)
    
    result = {}
    for name, config_data in configs_dict.items():
        result[name] = ModelInitConfig(
            name=name,
            requires_n_series=config_data.get('requires_n_series', False),
            supports_hist_exog=config_data.get('supports_hist_exog', True),
            supports_futr_exog=config_data.get('supports_futr_exog', False),
            supports_stat_exog=config_data.get('supports_stat_exog', False),
            special_init=config_data.get('special_init', False),
            custom_params=config_data.get('custom_params', {}),
            notes=config_data.get('notes', '')
        )
    
    return result

