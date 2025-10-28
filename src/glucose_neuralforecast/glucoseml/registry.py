"""Dataset registry loader and configuration models."""

from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
from pydantic import BaseModel, Field
from eliot import start_action


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration for a dataset."""
    
    timestamp_column: str
    glucose_column: Optional[str] = None
    glucose_column_alt: Optional[str] = None
    glucose_columns: Optional[List[str]] = None
    filter_event_type: Optional[str] = None
    subject_id_pattern: Optional[str] = None
    subject_range: Optional[List[int]] = None
    subject_id_from_folder: bool = False
    subject_id_from_file: bool = False
    multi_file_concat: bool = False
    file_pattern: Optional[str] = None
    exogenous_columns: Optional[List[str]] = None
    sensor_preference: Optional[List[str]] = None


class DatasetConfig(BaseModel):
    """Configuration for a single dataset."""
    
    name: str
    source: str
    format: str
    sensor_family: str
    sampling_minutes: int
    timezone: str
    license_note: str
    preprocessing: PreprocessingConfig


class ValidationConfig(BaseModel):
    """Validation configuration."""
    
    class GlucoseRange(BaseModel):
        min: float
        max: float
    
    class RateOfChange(BaseModel):
        min_time_delta_seconds: int
        max_rate_mg_dl_per_min: float
    
    class CoverageThreshold(BaseModel):
        daily_coverage_pct: float
        enabled: bool
    
    glucose_range: GlucoseRange
    rate_of_change: RateOfChange
    coverage_threshold: CoverageThreshold


class OutputSchemaConfig(BaseModel):
    """Output schema configuration."""
    
    class ColumnSpec(BaseModel):
        name: str
        type: str
        description: str
        optional: bool = False
    
    class MinimalPerSubject(BaseModel):
        columns: List[Dict[str, Any]]
    
    class UnifiedCombined(BaseModel):
        columns: List[Dict[str, Any]]
    
    minimal_per_subject: MinimalPerSubject
    unified_combined: UnifiedCombined


class RegistryConfig(BaseModel):
    """Full registry configuration."""
    
    datasets: Dict[str, DatasetConfig]
    validation: ValidationConfig
    output_schema: OutputSchemaConfig


def load_registry(registry_path: Optional[Path] = None) -> RegistryConfig:
    """
    Load dataset registry from YAML file.
    
    Args:
        registry_path: Path to registry.yaml file. If None, uses default location
                      in data/input/glucoseml/registry.yaml
    
    Returns:
        RegistryConfig: Parsed registry configuration
    """
    with start_action(action_type="load_registry", registry_path=str(registry_path)) as action:
        if registry_path is None:
            # Default to data/input/glucoseml/registry.yaml relative to project root
            from glucose_neuralforecast.utils import resolve_base_folder
            base = resolve_base_folder()
            registry_path = base / 'data' / 'input' / 'glucoseml' / 'registry.yaml'
        
        action.log(message_type="registry_path", path=str(registry_path))
        
        if not registry_path.exists():
            raise FileNotFoundError(f"Registry file not found: {registry_path}")
        
        with open(registry_path, 'r') as f:
            registry_data = yaml.safe_load(f)
        
        config = RegistryConfig(**registry_data)
        
        action.log(
            message_type="registry_loaded",
            num_datasets=len(config.datasets),
            datasets=list(config.datasets.keys())
        )
        
        return config


def get_dataset_config(dataset_name: str, registry_path: Optional[Path] = None) -> DatasetConfig:
    """
    Get configuration for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'CGMacros', 'BIG_IDEAS')
        registry_path: Optional path to registry file
    
    Returns:
        DatasetConfig: Configuration for the requested dataset
    
    Raises:
        KeyError: If dataset not found in registry
    """
    registry = load_registry(registry_path)
    
    if dataset_name not in registry.datasets:
        available = ', '.join(registry.datasets.keys())
        raise KeyError(f"Dataset '{dataset_name}' not found in registry. Available: {available}")
    
    return registry.datasets[dataset_name]


def list_datasets(registry_path: Optional[Path] = None) -> List[str]:
    """
    List all available datasets in the registry.
    
    Args:
        registry_path: Optional path to registry file
    
    Returns:
        List[str]: List of dataset names
    """
    registry = load_registry(registry_path)
    return list(registry.datasets.keys())

