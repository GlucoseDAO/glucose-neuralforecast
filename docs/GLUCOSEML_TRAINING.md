# Training on GlucoseML Datasets

This guide explains how to train NeuralForecast models on GlucoseML parquet datasets with automatic handling of multiple subjects and time series episodes.

## Overview

The GlucoseML datasets contain glucose data from multiple subjects, and each subject may have discontinuous measurement periods. The training system automatically:

1. **Loads multiple parquet files** (BIG_IDEAS, CGMacros, ShanghaiT1DM, ShanghaiT2DM, UCHTT1DM)
2. **Detects time gaps** in each subject's data
3. **Splits into episodes** based on configurable gap thresholds
4. **Creates unique series IDs** for each episode (e.g., `BIG_IDEAS__001__ep0`)
5. **Filters short episodes** that don't have enough data points

## Quick Start

### Basic Training on All Datasets

```bash
uv run train train --glucoseml
```

This loads all available GlucoseML datasets from `data/input/glucoseml/processed/` and trains the default models.

### Training on Specific Datasets

```bash
uv run train train --glucoseml --datasets 'BIG_IDEAS,ShanghaiT1DM'
```

### Custom Episode Segmentation

```bash
uv run glucose-forecast train \
  --glucoseml \
  --max-gap-minutes 120 \
  --min-episode-length 96
```

Parameters:
- `--max-gap-minutes`: Time gap threshold for splitting episodes (default: 60 minutes)
- `--min-episode-length`: Minimum data points required per episode (default: 48 points)

## Understanding Episode Segmentation

### How It Works

Given a subject with data like this:

```
BIG_IDEAS__001:
  2024-01-15 10:00:00  →  glucose: 125
  2024-01-15 10:05:00  →  glucose: 128
  ...
  2024-01-15 14:00:00  →  glucose: 142
  [GAP > 60 minutes]
  2024-01-16 08:00:00  →  glucose: 118
  2024-01-16 08:05:00  →  glucose: 120
```

The system creates:
- `BIG_IDEAS__001__ep0` (first measurement period)
- `BIG_IDEAS__001__ep1` (second measurement period after gap)

### Why Episode Segmentation?

1. **Prevents training issues**: Large time gaps can confuse models
2. **Improves predictions**: Models learn from continuous sequences
3. **Handles real data**: Real CGM data has natural breaks (sensor changes, charging, etc.)
4. **Matches NeuralForecast expectations**: Each `unique_id` should be a continuous series

## Data Format

### Input (GlucoseML Parquet)

The GlucoseML parquet files have this structure:

```
Columns:
- dataset: str (e.g., 'BIG_IDEAS')
- subject_id: str (e.g., '001')
- unique_id: str (e.g., 'BIG_IDEAS__001')
- ds: datetime (timestamp)
- y: float (glucose value in mg/dL)
- sensor_family: str
- sampling_minutes: int
- timezone: str
- Carb Value (grams): float (optional)
- Insulin Value (u): float (optional)
- Glucose Rate of Change (mg/dL/min): float (optional)
```

### Output (NeuralForecast Ready)

After processing:

```
Columns:
- unique_id: str (e.g., 'BIG_IDEAS__001__ep0')
- ds: datetime
- y: float

With exogenous (if --include-exogenous):
- carbs: float
- insulin: float
- glucose_rate: float
```

## Complete Examples

### Example 1: Quick Test with One Dataset

```bash
uv run glucose-forecast train \
  --glucoseml \
  --datasets 'BIG_IDEAS' \
  --models 'NHITS,LSTM' \
  --max-steps 500 \
  --horizon 12
```

### Example 2: Full Training Run

```bash
uv run glucose-forecast train \
  --glucoseml \
  --output-dir ./my_glucoseml_results \
  --run-id glucoseml_exp1 \
  --max-steps 2000 \
  --horizon 12 \
  --input-size 48
```

### Example 3: Specific Datasets with Custom Episodes

```bash
uv run glucose-forecast train \
  --glucoseml \
  --datasets 'BIG_IDEAS,CGMacros,ShanghaiT1DM' \
  --max-gap-minutes 90 \
  --min-episode-length 72 \
  --models 'NHITS,NBEATS,LSTM,TFT' \
  --horizon 18
```

### Example 4: Single Parquet File

```bash
uv run glucose-forecast train \
  --glucoseml \
  --data-file data/input/glucoseml/processed/BIG_IDEAS.parquet \
  --models 'NHITS' \
  --max-steps 1000
```

## Available Datasets

The GlucoseML processed directory contains:

1. **BIG_IDEAS.parquet** - CGM data from the BIG IDEAS Lab
2. **CGMacros.parquet** - CGM data with macronutrient information
3. **ShanghaiT1DM.parquet** - Type 1 diabetes patients from Shanghai
4. **ShanghaiT2DM.parquet** - Type 2 diabetes patients from Shanghai
5. **UCHTT1DM.parquet** - University of Chicago Type 1 diabetes data

## Programmatic Usage

### Loading Data Only

```python
from pathlib import Path
from glucose_neuralforecast.data import load_glucoseml_data

# Load all datasets
df = load_glucoseml_data(
    Path("data/input/glucoseml/processed"),
    max_gap_minutes=60,
    min_episode_length=48,
    include_exogenous=False
)

print(f"Loaded {len(df)} rows")
print(f"Number of episodes: {df['unique_id'].n_unique()}")
```

### Loading Specific Datasets

```python
df = load_glucoseml_data(
    Path("data/input/glucoseml/processed"),
    datasets=['BIG_IDEAS', 'ShanghaiT1DM'],
    max_gap_minutes=60,
    min_episode_length=48
)
```

### With Exogenous Variables

```python
df = load_glucoseml_data(
    Path("data/input/glucoseml/processed"),
    datasets=['BIG_IDEAS'],
    include_exogenous=True
)

print(f"Columns: {df.columns}")
# Output: ['unique_id', 'ds', 'y', 'carbs', 'insulin', 'glucose_rate']
```

## Parameters Reference

### Episode Segmentation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--max-gap-minutes` | int | 60 | Maximum time gap (minutes) before splitting into new episode |
| `--min-episode-length` | int | 48 | Minimum data points required to keep an episode |

**Choosing `--max-gap-minutes`:**
- **30-60 minutes**: Strict, more episodes, shorter continuous sequences
- **60-120 minutes**: Moderate, balanced episodes
- **120-240 minutes**: Lenient, fewer episodes, longer sequences (may include small gaps)

**Choosing `--min-episode-length`:**
- Must be ≥ `input_size` + `horizon`
- At 5-minute sampling: 48 points = 4 hours, 96 points = 8 hours
- Shorter episodes are automatically filtered out

### GlucoseML-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--glucoseml` | flag | False | Enable GlucoseML parquet data loading |
| `--datasets` | str | None | Comma-separated dataset names (e.g., 'BIG_IDEAS,CGMacros') |
| `--data-file` | str | None | Path to processed directory or specific parquet file |

## Comparison with Livia CSV Format

### Livia CSV (Original)

- Single subject with multiple measurement sequences
- Uses `sequence_id` to distinguish different measurement periods
- Data format: Wide format with many columns
- Exogenous variables: Insulin, carbs, flow_amount in specific columns

```bash
uv run glucose-forecast train --data-file data/input/livia_glucose.csv
```

### GlucoseML Parquet (New)

- Multiple subjects from different datasets
- Automatic episode detection based on time gaps
- Data format: Standardized parquet format
- Exogenous variables: Carbs, insulin, glucose_rate (when available)

```bash
uv run train train --glucoseml
```

## Troubleshooting

### Issue: "No parquet files found"

**Solution**: Check that processed parquet files exist:
```bash
ls data/input/glucoseml/processed/*.parquet
```

If missing, run preprocessing:
```bash
uv run glucoseml preprocess --all
```

### Issue: "No valid episodes after filtering"

**Cause**: All episodes are shorter than `min_episode_length`

**Solutions**:
1. Reduce `--min-episode-length`
2. Increase `--max-gap-minutes` to create longer episodes
3. Check if the dataset has enough continuous data

### Issue: Too many episodes

**Cause**: `max_gap_minutes` is too small, creating many short episodes

**Solution**: Increase `--max-gap-minutes` to tolerate larger gaps

### Issue: Episodes are discontinuous

**Cause**: `max_gap_minutes` is too large, keeping gaps within episodes

**Solution**: Decrease `--max-gap-minutes` for stricter segmentation

## Advanced Topics

### Memory Considerations

Loading all datasets simultaneously requires significant memory:

- All 5 datasets: ~200K+ rows
- Consider loading specific datasets for smaller memory footprint
- Use `--datasets` to limit scope

### Optimal Parameters for Different Goals

**For rapid iteration/testing:**
```bash
--datasets 'BIG_IDEAS' \
--max-steps 500 \
--models 'NHITS'
```

**For production-quality results:**
```bash
--glucoseml \
--max-steps 5000 \
--max-gap-minutes 60 \
--min-episode-length 96
```

**For research with exogenous variables:**
```bash
--glucoseml \
--datasets 'BIG_IDEAS,CGMacros' \
# Exogenous variables are included automatically per model
```

## See Also

- [GlucoseML Usage Guide](GLUCOSEML_USAGE.md) - Data preprocessing and ingestion
- [README.md](../README.md) - General training documentation
- [examples/train_glucoseml_example.py](../examples/train_glucoseml_example.py) - Complete examples

