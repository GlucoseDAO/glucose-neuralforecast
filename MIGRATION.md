# Migration to Versioned Training Runs

## Overview

The project now uses a versioned training run system where each training session creates an isolated directory with its own configuration and artifacts.

## What Changed

### Old Structure (deprecated)
```
data/output/
├── models/           # All models mixed together
├── metrics.csv      # Metrics from last training
└── cv_results_*.csv # CV results from last training
```

### New Structure
```
data/output/runs/
├── run_20241021_130601/     # Auto-generated run ID
│   ├── config.yaml          # Training configuration
│   ├── models/              # Models for this run
│   ├── metrics.csv          # Metrics for this run
│   ├── cv_results_*.csv     # CV results for this run
│   ├── predictions/         # Predictions from inference
│   └── plots/               # Generated plots
└── my_experiment_v1/        # Custom run ID
    └── ...
```

## New Features

### 1. Versioned Training Runs
Each training creates a unique run with timestamp-based ID:
```bash
# Auto-generated ID (e.g., run_20241021_130601)
uv run train

# Custom ID
uv run train --run-id my_experiment_v1
```

### 2. Configuration Persistence
Training configuration is automatically saved as `config.yaml` in each run directory.

### 3. List Training Runs
View all available training runs:
```bash
uv run list-runs
```

Output:
```
📊 Available Training Runs (2):

Run ID                    Models   Horizon  Steps    Status
================================================================================
run_20241021_143000      12       12       1000     📋 config 🤖 models
run_20241021_120000      8        12       500      📋 config 🤖 models
```

### 4. Inference with Run Selection
Select specific training run for inference:
```bash
uv run predict --run-id run_20241021_130601 --cherry-pick
```

## Migration Steps

### 1. Backup Old Data
Old training data has been backed up to:
```
data/old_output_backup_20251021_130601.tar.gz
```

### 2. Clean Start
The `data/output/` directory has been cleaned. Next training will create the new structure.

### 3. Update Your Workflow

**Old workflow:**
```bash
uv run train
uv run predict --cherry-pick
```

**New workflow:**
```bash
# Train (creates versioned run)
uv run train

# List runs
uv run list-runs

# Predict using specific run
uv run predict --run-id run_20241021_130601 --cherry-pick
```

## Benefits

1. **Experiment Tracking**: Each training run is isolated and tracked
2. **Reproducibility**: Configuration saved with each run
3. **Comparison**: Keep multiple training runs and compare results
4. **No Overwrites**: Previous runs are preserved
5. **Easy Rollback**: Switch between different trained models easily

## Backward Compatibility

Legacy mode is still supported (without `--run-id`), but not recommended:
```bash
# This will work but use old structure (not versioned)
uv run predict --cherry-pick
```

## Configuration File Updates

The `train_config.yaml` now includes `run_id`:
```yaml
run_id: null  # Auto-generated if null
data_file: null
output_dir: null
horizon: 12
input_size: 48
max_steps: 1000
models:
  - NBEATS
  - NHITS
  - LSTM
n_windows: 3
test_size: null
log_file: null
```

## New CLI Commands

### list-runs
Lists all available training runs with their status:
```bash
uv run list-runs [--output-dir DIR]
```

### train (updated)
Now creates versioned runs:
```bash
uv run train [--run-id ID] [other options...]
```

### predict (updated)
Now supports run selection:
```bash
uv run predict --run-id <run_id> [other options...]
```

## Example Workflow

```bash
# 1. Train models for experiment 1
uv run train --run-id experiment_baseline --max-steps 1000

# 2. Train models for experiment 2 with different settings
uv run train --run-id experiment_more_steps --max-steps 5000

# 3. List all runs
uv run list-runs

# 4. Compare results from experiment 1
uv run predict --run-id experiment_baseline --cherry-pick

# 5. Compare results from experiment 2
uv run predict --run-id experiment_more_steps --cherry-pick

# Both runs are preserved and can be compared!
```

## Questions?

- See `README.md` for full documentation
- Run `uv run train --help` or `uv run predict --help` for all options

