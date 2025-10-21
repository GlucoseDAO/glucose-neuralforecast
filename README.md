# glucose-neuralforecast

Repository to experiment with the NeuralForecast library for predicting glucose levels.

## Features

- **Iterative training**: Train models one-by-one with progress tracking
- **Crash resilience**: If one model fails, training continues with the next
- **Incremental results**: Metrics and plots saved after each model completes
- **Automatic visualization**: Generate prediction plots for each model
- **Comprehensive metrics**: MAE, MSE, RMSE, MAPE calculated for each model
- **Individual model saving**: Each model saved separately for easy deployment
- **Detailed logging**: Step numbers, progress tracking, and error logs
- **Wide default selection**: 12 diverse models trained by default
- **Structured logging**: Using Eliot for detailed execution tracking

## Installation

```bash
uv sync
```

## Usage

### List Available Models

To see all available models:

```bash
glucose-train list-models
```

### Basic Training

Train default 12 models (NBEATS, NHITS, NBEATSx, LSTM, GRU, DilatedRNN, TCN, BiTCN, DLinear, NLinear, TiDE, MLP):

```bash
glucose-train train
```

Or simply:

```bash
glucose-train
```

This will train all 12 models iteratively, showing progress (Step 1/12, Step 2/12, etc.) and saving results after each model completes.

### Custom Model Selection

Train specific models:

```bash
glucose-train train --models "NBEATS,NHITS,LSTM,GRU,MLP"
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
glucose-train train \
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
glucose-train train --models "PatchTST,iTransformer,Autoformer" --max-steps 500
```

Example with multiple model types:

```bash
glucose-train train --models "NBEATS,LSTM,TCN,DLinear,TFT" --horizon 24 --input-size 96
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

The data is automatically converted to NeuralForecast format with columns:
- `unique_id`: Sequence identifier
- `ds`: Datetime column
- `y`: Target values (glucose levels)

## Cross-Validation

The training uses NeuralForecast's cross-validation functionality:
- Data is automatically split into training and test sets
- Models are evaluated on multiple windows for robust performance assessment
- Step size is set to the forecast horizon for non-overlapping predictions
- Each model's predictions are saved for detailed analysis
