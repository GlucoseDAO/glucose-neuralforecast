# Quick Start Guide

## Installation

```bash
uv sync
```

## Quick Commands

### 1. List all available models (🔗 = supports exogenous variables)
```bash
glucose-train list-models
```

### 2. Generate default configuration file
```bash
glucose-train generate-config --output train_config.yaml
```

### 3. Train from configuration file (Recommended)
```bash
glucose-train train-from-config --config train_config.yaml
```

### 4. Train with command-line arguments
```bash
# Train default 12 models
glucose-train train

# Train specific models
glucose-train train --models "LSTM,GRU,TCN"

# Custom parameters
glucose-train train --models "NBEATS,NHITS" --horizon 24 --max-steps 2000
```

## Understanding Exogenous Variables

Models that support exogenous variables (marked with 🔗) are automatically trained twice:

1. **Univariate**: Using only glucose values
   - Example: `LSTM`
   
2. **Multivariate**: Using glucose + exogenous variables (insulin, carbs, flow_amount)
   - Example: `LSTM_exog`

### Example Output

When you train LSTM, GRU, and TCN:
```
Trained Models:
✓ LSTM (univariate - glucose only)
✓ LSTM_exog (multivariate - with insulin, carbs, etc.)
✓ GRU (univariate - glucose only)
✓ GRU_exog (multivariate - with insulin, carbs, etc.)
✓ TCN (univariate - glucose only)
✓ TCN_exog (multivariate - with insulin, carbs, etc.)
```

## Configuration File Template

```yaml
# train_config.yaml
data_file: null  # or "path/to/your/data.csv"
output_dir: null  # or "path/to/output"
horizon: 12
input_size: 48
max_steps: 1000
models:
  - NBEATS
  - NHITS
  - LSTM
  - GRU
  - TCN
  - BiTCN
  - DLinear
  - TiDE
n_windows: 3
test_size: null
log_file: null
```

## Models Supporting Exogenous Variables (23 total)

### Transformers (4)
- Autoformer
- FEDformer
- Informer
- VanillaTransformer

### CNNs (2)
- BiTCN
- TCN

### RNNs (7)
- DeepAR
- DeepNPTS
- DilatedRNN
- GRU
- LSTM
- RNN

### MLPs (6)
- HINT
- MLP
- MLPMultivariate
- NBEATSx
- NHITS

### Advanced (4)
- KAN
- TFT
- TiDE
- TimesNet
- TimeXer
- TSMixerx

## Output Structure

```
data/output/
├── models/
│   ├── LSTM/
│   ├── LSTM_exog/
│   ├── GRU/
│   ├── GRU_exog/
│   └── ...
├── plots/
│   ├── LSTM/
│   ├── LSTM_exog/
│   └── ...
├── metrics.csv                    # All model metrics
├── metrics_summary.csv            # Aggregated metrics
├── cv_results_LSTM.csv
├── cv_results_LSTM_exog.csv
└── training_summary.txt
```

## Common Workflows

### Workflow 1: Quick Test with Few Models
```bash
glucose-train train --models "LSTM,GRU" --max-steps 500
```

### Workflow 2: Production Training with Config
```bash
# 1. Generate config
glucose-train generate-config --output production_config.yaml

# 2. Edit production_config.yaml (increase max_steps, select models, etc.)

# 3. Train
glucose-train train-from-config --config production_config.yaml
```

### Workflow 3: Compare Exogenous vs Univariate
```bash
# Train models that support exogenous variables
glucose-train train --models "LSTM,GRU,TCN,TFT"

# Check metrics.csv to compare:
# - LSTM vs LSTM_exog
# - GRU vs GRU_exog
# - TCN vs TCN_exog
# - TFT vs TFT_exog
```

## Troubleshooting

### Issue: Model fails during training
- Check `data/output/error_ModelName.txt` for details
- Other models continue training automatically

### Issue: Want to skip exogenous training
- Currently, exogenous models are trained automatically
- You can ignore the `_exog` results if not needed

### Issue: Configuration file not found
```bash
# Generate a new one
glucose-train generate-config
```

## Next Steps

1. Generate your config: `glucose-train generate-config`
2. Edit the config to select your desired models
3. Train: `glucose-train train-from-config`
4. Analyze results in `data/output/metrics.csv`
5. Compare plots in `data/output/plots/`

