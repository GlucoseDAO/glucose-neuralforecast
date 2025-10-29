#!/usr/bin/env python3
"""
Example script for training on GlucoseML parquet data.

This demonstrates how to use the new GlucoseML data loading functionality
to train models on multiple subjects with automatic episode segmentation.
"""

from pathlib import Path
from glucose_neuralforecast.data import load_glucoseml_data
from glucose_neuralforecast.utils import resolve_base_folder


def main() -> None:
    """Load and explore GlucoseML data before training."""
    
    # Resolve project paths
    base = resolve_base_folder()
    data_dir = base / "data" / "input" / "glucoseml" / "processed"
    
    print("=" * 70)
    print("GlucoseML Data Loading Example")
    print("=" * 70)
    
    # Example 1: Load all datasets
    print("\n📊 Example 1: Loading all GlucoseML datasets")
    print("-" * 70)
    df_all = load_glucoseml_data(
        input_path=data_dir,
        max_gap_minutes=60,
        min_episode_length=48,
        include_exogenous=False
    )
    print(f"✅ Loaded {len(df_all)} rows")
    print(f"✅ Number of episodes (unique_id): {df_all['unique_id'].n_unique()}")
    print(f"✅ Columns: {df_all.columns}")
    print(f"✅ Date range: {df_all['ds'].min()} to {df_all['ds'].max()}")
    print("\nSample data:")
    print(df_all.head(5))
    
    # Example 2: Load specific datasets
    print("\n\n📊 Example 2: Loading specific datasets (BIG_IDEAS, ShanghaiT1DM)")
    print("-" * 70)
    df_specific = load_glucoseml_data(
        input_path=data_dir,
        datasets=['BIG_IDEAS', 'ShanghaiT1DM'],
        max_gap_minutes=60,
        min_episode_length=48,
        include_exogenous=False
    )
    print(f"✅ Loaded {len(df_specific)} rows")
    print(f"✅ Number of episodes: {df_specific['unique_id'].n_unique()}")
    
    # Example 3: Load with exogenous variables
    print("\n\n📊 Example 3: Loading with exogenous variables")
    print("-" * 70)
    df_exog = load_glucoseml_data(
        input_path=data_dir,
        datasets=['BIG_IDEAS'],
        max_gap_minutes=60,
        min_episode_length=48,
        include_exogenous=True
    )
    print(f"✅ Loaded {len(df_exog)} rows")
    print(f"✅ Number of episodes: {df_exog['unique_id'].n_unique()}")
    print(f"✅ Columns: {df_exog.columns}")
    
    # Example 4: Different gap thresholds
    print("\n\n📊 Example 4: Using different gap threshold (120 minutes)")
    print("-" * 70)
    df_larger_gap = load_glucoseml_data(
        input_path=data_dir,
        datasets=['BIG_IDEAS'],
        max_gap_minutes=120,  # Larger gaps allowed
        min_episode_length=48,
        include_exogenous=False
    )
    print(f"✅ Loaded {len(df_larger_gap)} rows")
    print(f"✅ Number of episodes: {df_larger_gap['unique_id'].n_unique()}")
    print(f"   (fewer episodes because larger gaps are tolerated)")
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
    
    print("\n📝 To train models on GlucoseML data, use:")
    print("\n   uv run glucose-forecast train --glucoseml")
    print("\n   Or for specific datasets:")
    print("\n   uv run glucose-forecast train --glucoseml --datasets 'BIG_IDEAS,ShanghaiT1DM'")
    print("\n   Or with custom parameters:")
    print("\n   uv run glucose-forecast train --glucoseml --max-gap-minutes 120 --min-episode-length 96")


if __name__ == "__main__":
    main()

