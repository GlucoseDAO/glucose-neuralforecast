# Changelog

## [Unreleased] - 2025-01-21

### Added

#### Exogenous Variables Support
- **Automatic dual training**: Models that support exogenous variables are now automatically trained twice:
  - Univariate version (glucose only) - e.g., `LSTM`
  - Multivariate version with exogenous variables - e.g., `LSTM_exog`
- **Exogenous variables included**:
  - `fast_insulin`: Fast-acting insulin doses
  - `long_insulin`: Long-acting insulin doses
  - `carbs`: Carbohydrate intake
  - `flow_amount`: Flow amount
- **Models supporting exogenous variables** (23 models):
  - Transformer-based: Autoformer, FEDformer, Informer, VanillaTransformer
  - CNN-based: BiTCN, TCN
  - RNN-based: DeepAR, DeepNPTS, DilatedRNN, GRU, LSTM, RNN
  - MLP-based: HINT, MLP, MLPMultivariate, NBEATSx, NHITS
  - Advanced: KAN, TFT, TiDE, TimesNet, TimeXer, TSMixerx

#### YAML Configuration Support
- **New command**: `glucose-train generate-config` - Generate default configuration file
- **New command**: `glucose-train train-from-config` - Train from YAML configuration
- **Configuration file features**:
  - Define all training parameters in one place
  - Version control friendly
  - Easy reproducibility
  - Default `train_config.yaml` included in repository
- **Configuration parameters**:
  - `data_file`: Path to input CSV
  - `output_dir`: Output directory
  - `horizon`: Forecast horizon
  - `input_size`: Historical window size
  - `max_steps`: Training steps per model
  - `models`: List of models to train
  - `n_windows`: Cross-validation windows
  - `test_size`: Test set size
  - `log_file`: Log file path

#### Enhanced Model Listing
- **Updated `list-models` command**: Now shows which models support exogenous variables with 🔗 marker
- Better categorization and display

### Changed
- **Data loading**: `load_glucose_data()` now accepts `include_exogenous` parameter
- **Training metrics**: Model names now include `_exog` suffix for exogenous runs
- **Output structure**: Separate directories and files for univariate and exogenous model versions
- **Dependencies**: Added `pydantic>=2.0.0` and `pyyaml>=6.0.0`

### Technical Details

#### New Files
- `src/glucose_neuralforecast/config.py`: Configuration management module
  - `TrainingConfig`: Pydantic model for configuration validation
  - `load_config()`: Load configuration from YAML
  - `save_default_config()`: Generate default configuration file
- `train_config.yaml`: Default training configuration file

#### Modified Files
- `src/glucose_neuralforecast/models.py`:
  - Added `get_models_supporting_exogenous()` function
  - Returns set of 23 models that support exogenous variables
  
- `src/glucose_neuralforecast/data.py`:
  - Updated `load_glucose_data()` with `include_exogenous` parameter
  - Automatic handling of exogenous columns with null-filling
  
- `src/glucose_neuralforecast/train.py`:
  - Added `generate_config` command
  - Added `train_from_config` command
  - Updated `list_models` command with exogenous indicators
  - Refactored training loop to support model configurations
  - Each model with exogenous support creates two configurations
  - Proper naming with `_exog` suffix for exogenous runs
  
- `src/glucose_neuralforecast/__init__.py`:
  - Exported new configuration functions
  - Exported `get_models_supporting_exogenous()`
  
- `pyproject.toml`:
  - Added `pydantic>=2.0.0` dependency
  - Added `pyyaml>=6.0.0` dependency
  - Added new script entry points
  
- `README.md`:
  - Comprehensive documentation of new features
  - YAML configuration examples
  - Exogenous variables documentation
  - Updated data format section

### Usage Examples

#### Generate and use configuration file:
```bash
# Generate default config
glucose-train generate-config --output my_config.yaml

# Edit my_config.yaml as needed

# Train from config
glucose-train train-from-config --config my_config.yaml
```

#### Train with automatic exogenous support:
```bash
# Train LSTM - will automatically create LSTM and LSTM_exog
glucose-train train --models "LSTM,GRU,TCN"

# Results in 6 trained models:
# - LSTM (univariate)
# - LSTM_exog (with exogenous variables)
# - GRU (univariate)
# - GRU_exog (with exogenous variables)
# - TCN (univariate)
# - TCN_exog (with exogenous variables)
```

### Migration Notes

For existing users:
1. Run `uv sync` to install new dependencies (`pydantic`, `pyyaml`)
2. Existing command-line usage remains unchanged
3. Models that don't support exogenous variables continue to work as before
4. Models with exogenous support now produce two sets of results automatically
5. Check `metrics.csv` for both base and `_exog` versions of supported models

