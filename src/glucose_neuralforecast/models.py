"""Model configuration and creation functions."""

from typing import Optional, List, Set

import typer
from neuralforecast.models import (
    NBEATS, NHITS, NBEATSx,
    LSTM, GRU, RNN,
    MLP, MLPMultivariate,
    DLinear, NLinear,
    TiDE, TCN, BiTCN,
    DeepAR, DeepNPTS,
    DilatedRNN,
    TFT, HINT,
    VanillaTransformer, Informer, Autoformer, FEDformer, PatchTST, iTransformer,
    StemGNN, SOFTS,
    TimesNet, TimeLLM, TimeMixer, TimeXer,
    TSMixer, TSMixerx,
    KAN, RMoK
)


def get_models_supporting_exogenous() -> Set[str]:
    """
    Get the set of model names that support exogenous variables (marked with F, F/H/S, or F/S in docs).

    Based on training tests, these models actually support historical exogenous variables:
    - Models that work: NHITS, NBEATSx, MLP, LSTM, GRU, RNN, DilatedRNN, TCN, BiTCN
    - Models that fail: VanillaTransformer, Informer, Autoformer, FEDformer

    Returns:
        Set[str]: Set of model names that support exogenous variables
    """
    return {
        # MLP-based models that support exogenous
        'NBEATSx',
        'NHITS',
        'MLP',
        'MLPMultivariate',

        # RNN-based models that support exogenous
        'LSTM',
        'GRU',
        'RNN',
        'DilatedRNN',

        # CNN-based models that support exogenous
        'TCN',
        'BiTCN',

        # Specialized models that support exogenous
        'TFT',
        'DeepAR',
        'DeepNPTS',
        'TiDE',
        'HINT',

        # Recent architectures that support exogenous
        'TimesNet',
        'TimeXer',
        'TSMixerx',

        # KAN models that support exogenous
        'KAN'
    }


def get_available_models(horizon: int, input_size: int, max_steps: int) -> dict:
    """
    Get dictionary of all available models with their constructors.
    
    Args:
        horizon: Forecast horizon
        input_size: Number of historical time steps
        max_steps: Maximum training steps
        
    Returns:
        dict: Dictionary mapping model names to their constructor functions
    """
    return {
        # MLP-based models
        'NBEATS': lambda: NBEATS(input_size=input_size, h=horizon, max_steps=max_steps),
        'NBEATSx': lambda: NBEATSx(input_size=input_size, h=horizon, max_steps=max_steps),
        'NHITS': lambda: NHITS(input_size=input_size, h=horizon, max_steps=max_steps),
        'MLP': lambda: MLP(input_size=input_size, h=horizon, max_steps=max_steps),
        'MLPMultivariate': lambda: MLPMultivariate(input_size=input_size, h=horizon, max_steps=max_steps),
        
        # RNN-based models
        'LSTM': lambda: LSTM(input_size=input_size, h=horizon, max_steps=max_steps),
        'GRU': lambda: GRU(input_size=input_size, h=horizon, max_steps=max_steps),
        'RNN': lambda: RNN(input_size=input_size, h=horizon, max_steps=max_steps),
        'DilatedRNN': lambda: DilatedRNN(input_size=input_size, h=horizon, max_steps=max_steps),
        
        # CNN-based models
        'TCN': lambda: TCN(input_size=input_size, h=horizon, max_steps=max_steps),
        'BiTCN': lambda: BiTCN(input_size=input_size, h=horizon, max_steps=max_steps),
        
        # Linear models
        'DLinear': lambda: DLinear(input_size=input_size, h=horizon, max_steps=max_steps),
        'NLinear': lambda: NLinear(input_size=input_size, h=horizon, max_steps=max_steps),
        
        # Transformer-based models
        'VanillaTransformer': lambda: VanillaTransformer(input_size=input_size, h=horizon, max_steps=max_steps),
        'Informer': lambda: Informer(input_size=input_size, h=horizon, max_steps=max_steps),
        'Autoformer': lambda: Autoformer(input_size=input_size, h=horizon, max_steps=max_steps),
        'FEDformer': lambda: FEDformer(input_size=input_size, h=horizon, max_steps=max_steps),
        'PatchTST': lambda: PatchTST(input_size=input_size, h=horizon, max_steps=max_steps),
        'iTransformer': lambda: iTransformer(input_size=input_size, h=horizon, max_steps=max_steps),
        
        # Specialized models
        'TFT': lambda: TFT(input_size=input_size, h=horizon, max_steps=max_steps),
        'DeepAR': lambda: DeepAR(input_size=input_size, h=horizon, max_steps=max_steps),
        'DeepNPTS': lambda: DeepNPTS(input_size=input_size, h=horizon, max_steps=max_steps),
        'TiDE': lambda: TiDE(input_size=input_size, h=horizon, max_steps=max_steps),
        'HINT': lambda: HINT(input_size=input_size, h=horizon, max_steps=max_steps),
        
        # GNN and advanced models
        'StemGNN': lambda: StemGNN(input_size=input_size, h=horizon, max_steps=max_steps),
        'SOFTS': lambda: SOFTS(input_size=input_size, h=horizon, max_steps=max_steps),
        
        # Recent/advanced architectures
        'TimesNet': lambda: TimesNet(input_size=input_size, h=horizon, max_steps=max_steps),
        'TimeLLM': lambda: TimeLLM(input_size=input_size, h=horizon, max_steps=max_steps),
        'TimeMixer': lambda: TimeMixer(input_size=input_size, h=horizon, max_steps=max_steps),
        'TimeXer': lambda: TimeXer(input_size=input_size, h=horizon, max_steps=max_steps),
        'TSMixer': lambda: TSMixer(input_size=input_size, h=horizon, max_steps=max_steps),
        'TSMixerx': lambda: TSMixerx(input_size=input_size, h=horizon, max_steps=max_steps),
        
        # KAN models
        'KAN': lambda: KAN(input_size=input_size, h=horizon, max_steps=max_steps),
        'RMoK': lambda: RMoK(input_size=input_size, h=horizon, max_steps=max_steps),
    }


def get_default_models() -> List[str]:
    """
    Get the default list of models to train.
    Only models that actually support exogenous variables are included here.

    Returns:
        List[str]: List of default model names that support exogenous variables
    """
    return [
        # MLP-based models that support exogenous
        'NHITS', 'NBEATSx', 'MLP', 'MLPMultivariate',
        # RNN-based models that support exogenous
        'LSTM', 'GRU', 'RNN', 'DilatedRNN',
        # CNN-based models that support exogenous
        'TCN', 'BiTCN',
        # Specialized models that support exogenous
        'TFT', 'DeepAR', 'DeepNPTS', 'TiDE', 'HINT',
        # Recent architectures that support exogenous
        'TimesNet', 'TimeXer', 'TSMixerx',
        # KAN models that support exogenous
        'KAN'
    ]


def get_model_list(
    horizon: int,
    input_size: int,
    max_steps: int = 1000,
    model_names: Optional[List[str]] = None
) -> List:
    """
    Create a list of models to train.
    
    Args:
        horizon: Forecast horizon
        input_size: Number of historical time steps
        max_steps: Maximum training steps
        model_names: List of model names to include. If None, uses a default set.
    
    Returns:
        List of initialized model objects
    """
    available_models = get_available_models(horizon, input_size, max_steps)
    
    if model_names is None:
        # Default wider selection of fast and reliable models
        model_names = get_default_models()
    
    models = []
    for name in model_names:
        if name in available_models:
            models.append(available_models[name]())
        else:
            typer.echo(f"Warning: Model '{name}' not available. Available models: {sorted(available_models.keys())}")
    
    return models


def get_models_by_category() -> dict:
    """
    Get models organized by category for display purposes.
    
    Returns:
        dict: Dictionary mapping category names to list of model names
    """
    return {
        "MLP-based models": ['NBEATS', 'NBEATSx', 'NHITS', 'MLP', 'MLPMultivariate'],
        "RNN-based models": ['LSTM', 'GRU', 'RNN', 'DilatedRNN'],
        "CNN-based models": ['TCN', 'BiTCN'],
        "Linear models": ['DLinear', 'NLinear'],
        "Transformer-based models": ['VanillaTransformer', 'Informer', 'Autoformer', 'FEDformer', 'PatchTST', 'iTransformer'],
        "Specialized models": ['TFT', 'DeepAR', 'DeepNPTS', 'TiDE', 'HINT'],
        "GNN and advanced models": ['StemGNN', 'SOFTS'],
        "Recent/advanced architectures": ['TimesNet', 'TimeLLM', 'TimeMixer', 'TimeXer', 'TSMixer', 'TSMixerx'],
        "KAN models": ['KAN', 'RMoK']
    }

