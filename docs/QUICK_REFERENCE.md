# Quick Reference Guide

## Training Commands

### Livia CSV (Single Subject)
```bash
# Default training
uv run train

# Custom parameters
uv run train --models "NHITS,LSTM" --horizon 12 --max-steps 2000
```

### GlucoseML Parquet (Multiple Subjects)
```bash
# All datasets
uv run train train --glucoseml

# Specific datasets
uv run train train --glucoseml --datasets 'BIG_IDEAS,ShanghaiT1DM'

# Custom episodes
uv run train train --glucoseml --max-gap-minutes 90 --min-episode-length 72

# Combined
uv run train train --glucoseml \
  --datasets 'BIG_IDEAS' \
  --models 'NHITS,LSTM,TFT' \
  --max-gap-minutes 120 \
  --min-episode-length 96 \
  --horizon 12 \
  --max-steps 2000
```

## Data Format

### NeuralForecast Required Columns
| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | str/int | Unique identifier for each time series |
| `ds` | datetime | Timestamp |
| `y` | float | Target value (glucose) |

### Episode Naming Convention
```
Format: <dataset>__<subject_id>__ep<episode_number>

Examples:
- BIG_IDEAS__001__ep0
- BIG_IDEAS__001__ep1
- ShanghaiT1DM__subj_15__ep0
```

## Key Parameters

### Episode Segmentation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-gap-minutes` | 60 | Max time gap before new episode |
| `--min-episode-length` | 48 | Min data points per episode |

### Model Training
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--horizon` | 12 | Forecast horizon (steps) |
| `--input-size` | 48 | Historical window size |
| `--max-steps` | 2000 | Training iterations |
| `--n-windows` | 3 | CV windows |

## Common Use Cases

### Quick Test
```bash
uv run train train --glucoseml \
  --datasets 'BIG_IDEAS' \
  --models 'NHITS' \
  --max-steps 500
```

### Production Run
```bash
uv run train train --glucoseml \
  --max-steps 5000 \
  --max-gap-minutes 60 \
  --min-episode-length 96
```

### Research with Exogenous
```bash
uv run train train --glucoseml \
  --datasets 'BIG_IDEAS,CGMacros' \
  # Exogenous vars auto-included per model
```

## Available Datasets

1. **BIG_IDEAS** - ~37K rows, 32 episodes
2. **CGMacros** - ~300K+ rows, many subjects
3. **ShanghaiT1DM** - Type 1 diabetes
4. **ShanghaiT2DM** - Type 2 diabetes  
5. **UCHTT1DM** - University of Chicago T1DM

## Exogenous Variables

### Livia CSV Format
- `fast_insulin`
- `long_insulin`
- `carbs`
- `flow_amount`

### GlucoseML Format
- `carbs` (from "Carb Value (grams)")
- `insulin` (from "Insulin Value (u)")
- `glucose_rate` (from "Glucose Rate of Change")

## Troubleshooting

### No parquet files found
```bash
# Check files exist
ls data/input/glucoseml/processed/*.parquet

# If missing, preprocess
uv run glucoseml preprocess --all
```

### Too many/few episodes
```bash
# More episodes (stricter)
--max-gap-minutes 30

# Fewer episodes (lenient)
--max-gap-minutes 120
```

### Episodes too short
```bash
# Lower minimum
--min-episode-length 24

# Or increase gap tolerance
--max-gap-minutes 120
```

## File Locations

```
glucose-neuralforecast/
├── data/
│   ├── input/
│   │   ├── livia_glucose.csv          # Single subject CSV
│   │   └── glucoseml/
│   │       └── processed/
│   │           ├── BIG_IDEAS.parquet
│   │           ├── CGMacros.parquet
│   │           ├── ShanghaiT1DM.parquet
│   │           ├── ShanghaiT2DM.parquet
│   │           └── UCHTT1DM.parquet
│   └── output/
│       └── runs/
│           └── <run_id>/
│               ├── config.yaml
│               ├── metrics.csv
│               └── models/
├── docs/
│   ├── GLUCOSEML_TRAINING.md    # Full guide
│   └── QUICK_REFERENCE.md       # This file
└── examples/
    └── train_glucoseml_example.py
```

## Next Steps

1. **Explore data**: `uv run python examples/train_glucoseml_example.py`
2. **Read guide**: [GLUCOSEML_TRAINING.md](GLUCOSEML_TRAINING.md)
3. **Train models**: `uv run train train --glucoseml`
4. **Check results**: `ls data/output/runs/<run_id>/`

