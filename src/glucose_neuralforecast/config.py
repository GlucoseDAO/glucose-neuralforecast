"""Configuration management for training."""

from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

import yaml
from pydantic import BaseModel, Field, ConfigDict


class TrainingConfig(BaseModel):
    """Configuration for neural forecast training."""
    
    model_config = ConfigDict(extra="forbid")
    
    run_id: Optional[str] = Field(
        None,
        description="Unique identifier for this training run. If not provided, a timestamp-based ID will be generated"
    )
    data_file: Optional[str] = Field(
        None,
        description="Path to the glucose CSV file"
    )
    output_dir: Optional[str] = Field(
        None,
        description="Directory to save model outputs"
    )
    horizon: int = Field(
        12,
        description="Forecast horizon (number of time steps to predict)"
    )
    input_size: int = Field(
        48,
        description="Input size for the model (number of historical time steps)"
    )
    max_steps: int = Field(
        1000,
        description="Maximum training steps for each model"
    )
    models: List[str] = Field(
        default_factory=lambda: [
            # MLP-based with exog support
            'NHITS', 'NBEATSx', 'MLP', 'MLPMultivariate',
            # RNN-based with exog support
            'LSTM', 'GRU', 'RNN', 'DilatedRNN',
            # CNN-based with exog support
            'TCN', 'BiTCN',
            # Transformer-based with exog support
            'VanillaTransformer', 'Informer', 'Autoformer', 'FEDformer',
            # Specialized models with exog support
            'TFT', 'DeepAR', 'DeepNPTS', 'TiDE', 'HINT',
            # Recent architectures with exog support
            'TimesNet', 'TimeXer', 'TSMixerx',
            # KAN models with exog support
            'KAN'
        ],
        description="List of models to train (all support exogenous variables)"
    )
    n_windows: int = Field(
        3,
        description="Number of cross-validation windows for evaluation"
    )
    test_size: Optional[int] = Field(
        None,
        description="Size of test set. If provided, n_windows is ignored"
    )
    log_file: Optional[str] = Field(
        None,
        description="Path to log file"
    )
    use_plotly: bool = Field(
        True,
        description="Use plotly for interactive plots (default: True). Set to False for matplotlib."
    )


def load_config(config_path: Path) -> TrainingConfig:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        TrainingConfig: Parsed configuration object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return TrainingConfig(**config_dict)


def save_default_config(output_path: Path) -> None:
    """
    Save a default configuration file.
    
    Args:
        output_path: Path where to save the default config
    """
    default_config = TrainingConfig()
    save_config(default_config, output_path)


def save_config(config: TrainingConfig, output_path: Path) -> None:
    """
    Save a training configuration to YAML file.
    
    Args:
        config: TrainingConfig object to save
        output_path: Path where to save the config
    """
    config_dict = {
        'run_id': config.run_id,
        'data_file': config.data_file,
        'output_dir': config.output_dir,
        'horizon': config.horizon,
        'input_size': config.input_size,
        'max_steps': config.max_steps,
        'models': config.models,
        'n_windows': config.n_windows,
        'test_size': config.test_size,
        'log_file': config.log_file,
        'use_plotly': config.use_plotly
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def generate_run_id() -> str:
    """
    Generate a unique run ID based on timestamp.
    
    Returns:
        str: Run ID in format 'run_YYYYMMDD_HHMMSS'
    """
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def list_training_runs(base_output_dir: Path) -> List[Dict[str, Any]]:
    """
    List all available training runs.
    
    Args:
        base_output_dir: Base output directory (e.g., data/output)
        
    Returns:
        List of dicts with run information (run_id, path, config_exists, models_exist)
    """
    runs_dir = base_output_dir / 'runs'
    
    if not runs_dir.exists():
        return []
    
    runs = []
    for run_path in sorted(runs_dir.iterdir(), reverse=True):
        if run_path.is_dir():
            config_path = run_path / 'config.yaml'
            models_path = run_path / 'models'
            
            run_info = {
                'run_id': run_path.name,
                'path': str(run_path),
                'config_exists': config_path.exists(),
                'models_exist': models_path.exists() and any(models_path.iterdir()) if models_path.exists() else False
            }
            
            # Try to load config to get additional info
            if config_path.exists():
                try:
                    config = load_config(config_path)
                    run_info['horizon'] = config.horizon
                    run_info['input_size'] = config.input_size
                    run_info['max_steps'] = config.max_steps
                    run_info['models'] = config.models
                except Exception:
                    pass
            
            runs.append(run_info)
    
    return runs


def get_latest_run(base_output_dir: Path) -> Optional[str]:
    """
    Get the most recent training run ID.
    
    Args:
        base_output_dir: Base output directory (e.g., data/output)
        
    Returns:
        str: The run_id of the most recent run, or None if no runs exist
    """
    runs = list_training_runs(base_output_dir)
    
    if not runs:
        return None
    
    # Runs are already sorted by most recent first
    # Only return runs that have models
    for run in runs:
        if run['models_exist']:
            return run['run_id']
    
    return None

