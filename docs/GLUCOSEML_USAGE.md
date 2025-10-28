# GlucoseML Dataset Integration

This module provides automated tools to download and preprocess open-access glucose monitoring datasets from the [GlucoseML repository](https://github.com/Diabetes-Datasets/GlucoseML_Diabetes_Datasets_NeurIPS2025-F5FC) into training-ready Parquet files with inline validation and covariates.

## Overview

The GlucoseML integration supports 5 open-access datasets:

| Dataset | Source | Sensor | Sampling | Subjects |
|---------|--------|--------|----------|----------|
| **BIG IDEAs** | PhysioNet | Dexcom | 5 min | 16 |
| **ShanghaiT1DM** | Figshare | FreeStyle Libre | 15 min | Variable |
| **ShanghaiT2DM** | Figshare | FreeStyle Libre | 15 min | Variable |
| **UCHTT1DM** | GitHub | Mixed | 5 min | Variable |
| **CGMacros** | PhysioNet | Dexcom/Libre | 5/15 min | Variable |

## Features

- **Simplified Pipeline**: One-step preprocessing from raw data to parquet with inline validation
- **Data Quality**: Built-in validation with configurable thresholds:
  - Glucose range filtering (40-400 mg/dL)
  - Rate-of-change limits (≤20 mg/dL/min)
  - Daily coverage requirements (≥70% expected readings)
- **Exogenous Covariates**: Automatically include available covariates (insulin, carbs, heart rate, etc.)
- **Efficient Storage**: One Parquet file per dataset with all subjects
- **Per-Dataset Stats**: Automatic generation of statistics (glucose values, subjects, covariates)
- **Unified Schema**: Standardized format compatible with NeuralForecast

## Installation

The GlucoseML integration is included in the main package. Ensure all dependencies are installed:

```bash
uv sync
```

## Quick Start

### Complete Pipeline

Run the entire pipeline for all datasets (download + preprocess to parquet):

```bash
glucoseml pipeline --datasets all
```

Or for specific datasets:

```bash
glucoseml pipeline --datasets "BIG_IDEAS,CGMacros"
```

Skip download if data is already available:

```bash
glucoseml pipeline --datasets all --skip-download
```

### Individual Steps

#### 1. List Available Datasets

```bash
glucoseml list-available
```

#### 2. Download Raw Data

```bash
# Download all datasets
glucoseml download --datasets all

# Download specific datasets
glucoseml download --datasets "BIG_IDEAS,ShanghaiT1DM"

# Custom output directory
glucoseml download --datasets all --output /path/to/raw/data
```

#### 3. Preprocess to Parquet (with Validation)

Converts raw data to standardized parquet format with inline validation:

```bash
# With validation (default)
glucoseml preprocess --datasets all

# Skip validation
glucoseml preprocess --datasets all --no-validation

# Custom paths
glucoseml preprocess --datasets all \
    --input /path/to/raw \
    --output /path/to/output
```

This single command:
- Reads raw data
- Applies validation filters (glucose range, rate-of-change, coverage)
- Includes exogenous covariates
- Writes one Parquet file per dataset
- Generates statistics (CSV + YAML)

## Directory Structure

After running the pipeline, your data directory will contain:

```
data/input/glucoseml/
├── registry.yaml              # Dataset configuration
├── raw/                       # Downloaded raw data
│   ├── BIG_IDEAS/
│   │   ├── Dexcom_001.csv
│   │   ├── Dexcom_002.csv
│   │   └── ...
│   ├── ShanghaiT1DM/
│   ├── ShanghaiT2DM/
│   ├── UCHTT1DM/
│   └── CGMacros/
├── stats/                     # Statistics
│   ├── per_dataset_counts.csv
│   └── stats.yaml
├── BIG_IDEAS.parquet         # Per-dataset parquet files
├── ShanghaiT1DM.parquet
├── ShanghaiT2DM.parquet
├── UCHTT1DM.parquet
└── CGMacros.parquet
```

## Data Schema

### Parquet Schema

Each dataset parquet file contains:

**Standard Columns:**
- `dataset`: Dataset name (e.g., "BIG_IDEAS")
- `subject_id`: Dataset-native subject identifier
- `unique_id`: Global identifier (`{dataset}__{subject_id}`)
- `ds`: Datetime (NeuralForecast standard)
- `y`: Blood glucose value (NeuralForecast standard)
- `sensor_family`: Sensor type (Dexcom, Libre, etc.)
- `sampling_minutes`: Expected sampling cadence (5 or 15)
- `timezone`: Timezone policy

**Exogenous Covariates (dataset-specific):**
- **BIG_IDEAS**: `Insulin Value (u)`, `Carb Value (grams)`, `Duration (hh:mm:ss)`, `Glucose Rate of Change (mg/dL/min)`
- **Shanghai T1/T2DM**: `CBG (mg / dl)`, `Blood Ketone (mmol / L)`, insulin doses, dietary intake, etc.
- **UCHTT1DM**: `Carbs (g)`, `IGAR (g)`, `Steps`, `Heart Rate (bpm)`, `Value_interp (mg/dl)`
- **CGMacros**: `HR`, `Calories`, `Carbs`, `Protein`, `Fat`, `Meal Type`, etc.

Note: Covariates retain dataset-specific names to preserve context and avoid conflicts.

## Validation Rules

Validation is applied inline during preprocessing (configured in `registry.yaml`):

### Glucose Range
- **Min**: 40 mg/dL
- **Max**: 400 mg/dL
- **Action**: Drop values outside range

### Rate of Change
- **Min time delta**: 30 seconds (drop consecutive readings < 30s apart)
- **Max rate**: 20 mg/dL/min (drop if |ΔBG/Δt| exceeds limit)

### Coverage Threshold
- **Daily coverage**: 70% of expected readings per day
- **Action**: Drop entire days below threshold
- **Calculation**: Based on sensor-specific sampling rate (5 or 15 minutes)

## Statistics

After preprocessing, check the generated statistics:

```bash
# View per-dataset counts (CSV)
cat data/input/glucoseml/stats/per_dataset_counts.csv

# View detailed stats (YAML)
cat data/input/glucoseml/stats/stats.yaml
```

The stats include:
- Total glucose values per dataset
- Number of subjects
- Date range
- Covariates available
- Drop rate from validation

## Configuration

Edit `data/input/glucoseml/registry.yaml` to customize:

- Dataset sources and formats
- Preprocessing parameters (column names, filters)
- Exogenous covariates to include
- Validation thresholds
- Timezone handling

Example configuration snippet:

```yaml
validation:
  glucose_range:
    min: 40
    max: 400
  rate_of_change:
    min_time_delta_seconds: 30
    max_rate_mg_dl_per_min: 20
  coverage_threshold:
    daily_coverage_pct: 70
    enabled: true

datasets:
  BIG_IDEAS:
    preprocessing:
      timestamp_column: "Time"
      glucose_column: "Glucose Value (mg/dL)"
      exogenous_columns:
        - "Insulin Value (u)"
        - "Carb Value (grams)"
        - "Glucose Rate of Change (mg/dL/min)"
```

## Training with GlucoseML Data

Once data is preprocessed, load the parquet files directly:

```python
import polars as pl

# Load a single dataset
df = pl.read_parquet("data/input/glucoseml/BIG_IDEAS.parquet")

# Load multiple datasets
df_all = pl.concat([
    pl.read_parquet("data/input/glucoseml/BIG_IDEAS.parquet"),
    pl.read_parquet("data/input/glucoseml/CGMacros.parquet")
])

# Train with NeuralForecast
# Select standard columns for basic training
df_train = df_all.select(["unique_id", "ds", "y"])

# Or include covariates for advanced models
df_train_cov = df_all.select([
    "unique_id", "ds", "y",
    # Dataset-specific covariates (fill nulls for missing datasets)
    "Insulin Value (u)",  # BIG_IDEAS
    "Carbs (g)",          # UCHTT1DM
    "HR"                  # CGMacros
])
```

## Advanced Usage

### Programmatic API

Use the modules directly in Python:

```python
from pathlib import Path
from glucose_neuralforecast.glucoseml import load_registry
from glucose_neuralforecast.glucoseml.preprocess import preprocess_all_datasets

# Load registry
registry = load_registry()

# Run preprocessing with validation
datasets = ["BIG_IDEAS", "CGMacros"]
raw_dir = Path("data/input/glucoseml/raw")
output_dir = Path("data/input/glucoseml")

stats = preprocess_all_datasets(
    datasets,
    raw_dir,
    output_dir,
    registry_config=registry,
    apply_validation=True
)

# Check stats
for dataset, dataset_stats in stats.items():
    print(f"{dataset}:")
    print(f"  Subjects: {dataset_stats['subjects']}")
    print(f"  Glucose values: {dataset_stats['total_glucose_values']}")
    print(f"  Drop rate: {dataset_stats['drop_rate'] * 100:.2f}%")
    print(f"  Covariates: {', '.join(dataset_stats['covariates'])}")
```

### Custom Validation Settings

Disable validation or modify thresholds:

```python
# Skip validation
stats = preprocess_all_datasets(
    datasets,
    raw_dir,
    output_dir,
    registry_config=registry,
    apply_validation=False
)

# Or modify registry config before preprocessing
registry.validation.coverage_threshold.daily_coverage_pct = 80
stats = preprocess_all_datasets(
    datasets,
    raw_dir,
    output_dir,
    registry_config=registry,
    apply_validation=True
)
```

## Dataset-Specific Notes

### BIG IDEAs (PhysioNet)
- **License**: Open Data Commons Open Database License v1.0
- **Subjects**: 16 subjects (001-016)
- **Files**: One CSV per subject (`Dexcom_NNN.csv`)
- **Covariates**: Insulin, carbs, duration, rate of change

### Shanghai T1DM / T2DM (Figshare)
- **License**: No explicit license (local use only)
- **Format**: Excel files, may span multiple files per subject
- **Preprocessing**: Concatenate files by subject ID (prefix before underscore)
- **Covariates**: CBG, blood ketone, insulin doses, dietary intake

### UCHTT1DM (GitHub)
- **License**: No explicit license (local use only)
- **Source**: Git clone from GitHub repository
- **Structure**: Subject folders with multiple Excel files (Glucose, Carbohydrates, Heart Rate, etc.)
- **Preprocessing**: Joins multiple Excel files on timestamp per subject
- **Covariates**: Carbs, IGAR, steps, heart rate, interpolated glucose

### CGMacros (PhysioNet)
- **License**: Open Data Commons Attribution License v1.0
- **Sensors**: Both Dexcom and FreeStyle Libre
- **Preprocessing**: Prefers Dexcom GL when both available
- **Covariates**: Heart rate, calories, macronutrients (carbs, protein, fat), meal type

## Troubleshooting

### Download Issues

If downloads fail:

1. **PhysioNet**: Check network connectivity and ensure wget is installed
2. **Figshare**: Rate limiting may apply; retry after a delay
3. **GitHub**: Ensure git is installed for UCHTT1DM

### Preprocessing Errors

Common issues:

- **Excel engine errors**: Ensure `openpyxl` is installed (included in dependencies)
- **Column not found**: Check dataset format matches registry configuration
- **Empty output**: Some subjects may have no valid data after validation

### High Drop Rates

Check the drop rate in stats. High drop rates (>50%) may indicate:
- Coverage threshold too strict (use `--no-validation` or adjust registry)
- Sensor issues in source data
- Timezone/timestamp parsing problems

## Deprecated Commands

The following commands are deprecated and will show warnings:
- `glucoseml validate` - Use `preprocess` instead (includes inline validation)
- `glucoseml combine` - Use `preprocess` instead (writes parquets directly)

## License and Attribution

This integration implements the preprocessing pipeline from:

> **GlucoseML: A Comprehensive Public Benchmark for Continuous Glucose Monitoring Datasets**  
> NeurIPS 2025 Datasets and Benchmarks Track

The GlucoseML datasets have various licenses:
- **BIG IDEAs, CGMacros**: Open Data Commons licenses (see individual datasets)
- **Shanghai T1DM/T2DM, UCHTT1DM**: No explicit licenses; use for local research only

Always check the license notes in `registry.yaml` and cite the original dataset sources when publishing results.

## See Also

- [Main Quickstart Guide](QUICKSTART.md)
- [GlucoseML Integration Plan](GLUCOSEML_INTEGRATION.md)
- [Plotly Visualization Guide](PLOTLY_VISUALIZATION.md)
