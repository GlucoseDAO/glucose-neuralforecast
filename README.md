# glucose-neuralforecast

Repository to experiment with the NeuralForecast library for predicting glucose levels.

## Features

### Training
- **Versioned training runs**: Each training run gets a unique ID and isolated directory structure
- **Configuration persistence**: Training configuration automatically saved with each run
- **Run management**: List and select from multiple training runs for inference
- **Exogenous variables support**: Models that support exogenous variables are automatically trained both with and without them
- **YAML configuration**: Define training parameters in a YAML file for reproducibility
- **Iterative training**: Train models one-by-one with progress tracking
- **Crash resilience**: If one model fails, training continues with the next
- **Incremental results**: Metrics and plots saved after each model completes
- **Automatic visualization**: Generate prediction plots for each model
- **Comprehensive metrics**: MAE, MSE, RMSE, MAPE calculated for each model
- **Individual model saving**: Each model saved separately for easy deployment
- **Detailed logging**: Step numbers, progress tracking, and error logs
- **Wide default selection**: 12 diverse models trained by default
- **Structured logging**: Using Eliot for detailed execution tracking

### Inference
- **Model loading**: Load and use any trained model for predictions
- **Cherry-pick mode**: Automatically select best or random sequences for evaluation
- **Multi-model comparison**: Compare predictions from multiple models simultaneously
- **Comparison plots**: Generate professional visualizations comparing all models
- **Flexible prediction**: Predict on specific sequences or entire datasets
- **Batch processing**: Run inference with multiple models in parallel
- **Model selection**: Choose specific models or use all trained models

## Installation

```bash
uv sync
```

## Usage

### Quick Start

1. **List available models**:
```bash
uv run list-models
```

2. **Train models** (automatically creates a versioned run):
```bash
uv run train
```

3. **List training runs**:
```bash
uv run list-runs
```

4. **Run inference** on a specific training run:
```bash
uv run predict --run-id run_20241021_130000 --cherry-pick
```

### Training Runs Management

Each training session creates a unique run with its own directory:
- **Structure**: `data/output/runs/<run_id>/`
- **Run ID**: Automatically generated timestamp (e.g., `run_20241021_130601`) or custom via `--run-id`
- **Contents**: models/, config.yaml, metrics.csv, cv_results_*.csv

List all training runs:
```bash
uv run list-runs
```

Output example:
```
📊 Available Training Runs (2):

Run ID                    Models   Horizon  Steps    Status
================================================================================
run_20241021_143000      12       12       1000     📋 config 🤖 models
run_20241021_120000      8        12       500      📋 config 🤖 models

💡 Use --run-id to specify a run for inference
   Example: predict --run-id run_20241021_143000
```

### List Available Models

To see all available models (models with 🔗 support exogenous variables):

```bash
uv run list-models
```

### Training with YAML Configuration (Recommended)

Generate a default configuration file:

```bash
generate-config --output train_config.yaml
```

Edit the generated `train_config.yaml` file to customize training parameters, then train:

```bash
train-from-config --config train_config.yaml
```

Example `train_config.yaml`:

```yaml
run_id: null  # Auto-generated timestamp if null (e.g., run_20241021_130601)
data_file: null  # Uses default data/input/livia_glucose.csv
output_dir: null  # Uses default data/output
horizon: 12
input_size: 48
max_steps: 1000
models:
- NBEATS
- NHITS
- LSTM
- GRU
- TCN
n_windows: 3
test_size: null
log_file: null
```

### Basic Training (Command Line)

Train default 12 models (automatically creates a versioned run):

```bash
uv run train
```

Train with custom run ID:

```bash
uv run train --run-id my_experiment_v1
```

Train specific models:

```bash
uv run train --models "NBEATS,NHITS,LSTM" --max-steps 2000
```

This will train models iteratively, showing progress and saving results after each model completes.
Each training run is saved in `data/output/runs/<run_id>/` with its own config.yaml.

**Note**: Models that support exogenous variables will be automatically trained twice:
- Once as univariate (glucose only) - e.g., `LSTM`
- Once with exogenous variables (insulin, carbs, flow_amount) - e.g., `LSTM_exog`

### Custom Model Selection

Train specific models:

```bash
train --models "NBEATS,NHITS,LSTM,GRU,MLP"
```

### Available Models

**MLP-based models:**
- NBEATS - Neural Basis Expansion Analysis
- NBEATSx - NBEATS with exogenous variables
- NHITS - Neural Hierarchical Interpolation for Time Series
- MLP - Multi-Layer Perceptron
- MLPMultivariate - Multivariate MLP

**RNN-based models:**
- LSTM - Long Short-Term Memory
- GRU - Gated Recurrent Unit
- RNN - Recurrent Neural Network
- DilatedRNN - Dilated Recurrent Neural Network

**CNN-based models:**
- TCN - Temporal Convolutional Network
- BiTCN - Bidirectional Temporal Convolutional Network

**Linear models:**
- DLinear - Decomposition Linear
- NLinear - Normalization Linear

**Transformer-based models:**
- VanillaTransformer - Standard Transformer
- Informer - Informer: Beyond Efficient Transformer
- Autoformer - Autoformer with Auto-Correlation
- FEDformer - Frequency Enhanced Decomposed Transformer
- PatchTST - Patch Time Series Transformer
- iTransformer - Inverted Transformer

**Specialized models:**
- TFT - Temporal Fusion Transformer
- DeepAR - Deep Autoregressive
- DeepNPTS - Deep Neural Point Time Series
- TiDE - Time-series Dense Encoder
- HINT - Hierarchical Interpolation Network for Time series

**GNN and advanced models:**
- StemGNN - Spectral Temporal Graph Neural Network
- SOFTS - Self-Organizing Fuzzy Time Series

**Recent/advanced architectures:**
- TimesNet - TimesNet with Period Detection
- TimeLLM - Time Series with Large Language Models
- TimeMixer - Time Series Mixing
- TimeXer - Time Series Cross-series
- TSMixer - Time Series Mixer
- TSMixerx - TSMixer with exogenous variables

**KAN models:**
- KAN - Kolmogorov-Arnold Networks
- RMoK - Recurrent Mixture of KANs

### Advanced Options

```bash
train \
  --data-file data/input/livia_glucose.csv \
  --output-dir data/output \
  --horizon 12 \
  --input-size 48 \
  --max-steps 1000 \
  --models "NBEATS,NHITS,LSTM" \
  --n-windows 3 \
  --test-size 100
```

Example with transformer models:

```bash
train --models "PatchTST,iTransformer,Autoformer" --max-steps 500
```

Example with multiple model types:

```bash
train --models "NBEATS,LSTM,TCN,DLinear,TFT" --horizon 24 --input-size 96
```

### Command Line Arguments

- `--data-file, -d`: Path to glucose CSV file (default: `data/input/livia_glucose.csv`)
- `--output-dir, -o`: Directory to save outputs (default: `data/output`)
- `--horizon, -h`: Forecast horizon in time steps (default: 12)
- `--input-size, -i`: Number of historical time steps to use (default: 48)
- `--max-steps, -s`: Maximum training steps per model (default: 1000)
- `--models, -m`: Comma-separated list of models to train
- `--n-windows, -n`: Number of cross-validation windows (default: 3)
- `--test-size, -t`: Size of test set (overrides n-windows if provided)
- `--log-file, -l`: Path to log file (default: `data/output/training.log`)

## Output Structure

After training, the following files are generated:

```
data/output/
├── models/                           # Individual model directories
│   ├── NBEATS/
│   │   ├── configuration.pkl
│   │   ├── dataset.pkl
│   │   └── NBEATS_0.ckpt
│   ├── NHITS/
│   ├── LSTM/
│   └── ... (one directory per trained model)
├── plots/                            # Prediction visualizations (using utilsforecast)
│   ├── NBEATS/
│   │   ├── sequence_0.png
│   │   ├── sequence_1.png
│   │   └── sequence_2.png
│   ├── NHITS/
│   └── ... (plots for each model)
├── metrics.csv                       # Model performance metrics (updated after each model)
├── cv_results_NBEATS.csv            # Cross-validation results per model
├── cv_results_NHITS.csv
├── ... (one CV file per model)
├── error_ModelName.txt               # Error logs for failed models (if any)
├── training_summary.txt              # Final summary report
└── training.log                      # Structured eliot logs
```

### Incremental Saving

Results are saved after each model completes:
- ✅ If one model fails, you still have results from successful models
- ✅ Metrics CSV is updated incrementally
- ✅ Each model's CV results and plots are saved immediately
- ✅ Progress is visible with step numbers (Step 3/12, etc.)

## Metrics

The following metrics are calculated for each model:

- **MAE** (Mean Absolute Error): Average absolute difference between predictions and actual values
- **MSE** (Mean Squared Error): Average squared difference between predictions and actual values
- **RMSE** (Root Mean Squared Error): Square root of MSE
- **MAPE** (Mean Absolute Percentage Error): Average absolute percentage error

## Data Format

Input CSV should contain:
- `sequence_id`: Identifier for time series sequences
- `Timestamp (YYYY-MM-DDThh:mm:ss)`: Timestamp column
- `Event Type`: Type of event (filtered to 'EGV' for glucose values)
- `Glucose Value (mg/dL)`: Target glucose measurements
- `Fast-Acting Insulin Value (u)`: Fast-acting insulin doses (optional, for exogenous models)
- `Long-Acting Insulin Value (u)`: Long-acting insulin doses (optional, for exogenous models)
- `Carb Value (grams)`: Carbohydrate intake (optional, for exogenous models)
- `flow_amount`: Flow amount (optional, for exogenous models)

The data is automatically converted to NeuralForecast format:

**Univariate format (base models):**
- `unique_id`: Sequence identifier
- `ds`: Datetime column
- `y`: Target values (glucose levels)

**Multivariate format (exogenous models):**
- `unique_id`: Sequence identifier
- `ds`: Datetime column
- `y`: Target values (glucose levels)
- `fast_insulin`: Fast-acting insulin (filled with 0 for missing values)
- `long_insulin`: Long-acting insulin (filled with 0 for missing values)
- `carbs`: Carbohydrate intake (filled with 0 for missing values)
- `flow_amount`: Flow amount (filled with 0 for missing values)

## Cross-Validation

The training uses NeuralForecast's cross-validation functionality:
- Data is automatically split into training and test sets
- Models are evaluated on multiple windows for robust performance assessment
- Step size is set to the forecast horizon for non-overlapping predictions
- Each model's predictions are saved for detailed analysis

## Inference

After training models, you can use them for prediction and comparison.

### List Available Trained Models

See which models have been trained:

```bash
list-trained
```

### Run Predictions

#### Predict with Cherry-Pick Mode (Best Sequence)

Automatically select the sequence with the best MAE and compare all trained models:

```bash
uv run predict --run-id run_20241021_130601 --cherry-pick
```

This will:
1. Load models from the specified training run
2. Find the sequence (unique_id) with the best average MAE across all models
3. Run inference with all trained models on that sequence
4. Save predictions to CSV files
5. Generate a comparison plot showing all model predictions

#### Predict with Random Sequence

Select a random sequence for comparison:

```bash
uv run predict --run-id run_20241021_130601 --cherry-pick --random
```

#### Predict on Specific Sequence

Predict on a specific sequence by unique_id:

```bash
uv run predict --run-id run_20241021_130601 --unique-id "3"
```

#### Predict with Selected Models Only

Use only specific models for prediction:

```bash
uv run predict --run-id run_20241021_130601 --cherry-pick --models "NBEATS,NHITS,LSTM"
```

#### Run Without Plotting

Skip plot generation (useful for large-scale predictions):

```bash
uv run predict --run-id run_20241021_130601 --cherry-pick --no-plot
```

#### Save Predictions Without Plots

Save predictions but don't generate comparison plots:

```bash
uv run predict --run-id run_20241021_130601 --cherry-pick --no-save-predictions
```

### Inference Command Options

- `--data-file, -d`: Path to glucose CSV file (default: `data/input/livia_glucose.csv`)
- `--output-dir, -o`: Base output directory (default: `data/output`)
- `--run-id, -r`: Training run ID to use (e.g., `run_20241021_130601`). If not provided, uses legacy mode
- `--models, -m`: Comma-separated list of models to use for inference
- `--cherry-pick, -c`: Enable cherry-pick mode to select a specific sequence
- `--best/--random`: In cherry-pick mode, select best MAE sequence or random
- `--unique-id, -u`: Specific unique_id to predict (overrides cherry-pick)
- `--seed, -s`: Random seed for reproducibility with random selection
- `--save-predictions/--no-save-predictions`: Save predictions to CSV (default: True)
- `--plot/--no-plot`: Generate comparison plot (default: True)

### Inference Output

Inference generates the following outputs within the run directory:

```
data/output/runs/<run_id>/
├── config.yaml                      # Training configuration
├── models/                          # Trained model checkpoints
├── metrics.csv                      # Evaluation metrics
├── cv_results_*.csv                 # Cross-validation results
├── predictions/                     # Prediction CSV files
│   ├── predictions_NBEATS.csv
│   ├── predictions_NHITS.csv
│   ├── predictions_LSTM.csv
│   └── ... (one file per model)
└── plots/
    └── comparison/                  # Model comparison plots
        └── comparison_<unique_id>.png
```

### Usage Examples

**Example 1: Quick comparison of best sequence from specific run**

```bash
# List available training runs
uv run list-runs

# Find best sequence and compare all models from a specific run
uv run predict --run-id run_20241021_130601 --cherry-pick
```

**Example 2: Compare specific models on best sequence**

```bash
# Compare only NBEATS, LSTM, and TCN on best sequence
uv run predict --run-id run_20241021_130601 --cherry-pick --models "NBEATS,LSTM,TCN"
```

**Example 3: Reproducible random selection**

```bash
# Randomly select sequence with fixed seed
uv run predict --run-id run_20241021_130601 --cherry-pick --random --seed 42
```

**Example 4: Predict on specific patient data**

```bash
# Predict on specific unique_id
uv run predict --run-id run_20241021_130601 --unique-id "3" --models "NHITS,LSTM,GRU"
```

**Example 5: Batch predictions without plots**

```bash
# Generate predictions for all sequences without plotting (legacy mode)
uv run predict --no-plot
```

### Cherry-Pick Mode

Cherry-pick mode helps you quickly evaluate and compare model performance:

- **Best mode (default)**: Selects the sequence with the lowest average MAE across all models
  - Useful for seeing how models perform on "easy" sequences
  - Good for validating that models learned meaningful patterns
  
- **Random mode**: Randomly selects a sequence
  - Useful for unbiased evaluation
  - Can reveal how models handle different types of patterns
  - Use `--seed` for reproducibility

### Comparison Plots

The comparison plots generated by inference show:
- **Historical data**: Previous glucose values (up to 500 points)
- **Predictions**: Forecast from each selected model
- **Multiple models**: All models overlaid for easy comparison
- **Time series continuity**: Seamless transition from history to forecast

Plots use the `utilsforecast.plotting.plot_series` function for professional visualization.

## Example Scripts

### Programmatic Inference Example

See `examples/inference_example.py` for a complete example of using the inference module programmatically:

```python
from glucose_neuralforecast.inference import (
    get_available_trained_models,
    cherry_pick_sequence,
    run_inference,
    plot_model_comparison,
)

# Get available models
available_models = get_available_trained_models(models_dir)

# Cherry-pick best sequence
selected_id = cherry_pick_sequence(metrics_path, best=True)

# Run inference
predictions = run_inference(df_filtered, models_to_use, models_dir)

# Generate comparison plot
plot_model_comparison(df_pandas, predictions, selected_id, plots_dir)
```

Run the example:

```bash
cd examples
python inference_example.py
```

## Development

After making changes to dependencies:

```bash
uv sync
```

Install development dependencies:

```bash
uv sync --group dev
```
