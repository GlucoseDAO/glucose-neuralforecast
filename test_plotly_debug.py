#!/usr/bin/env python3
"""Debug script to test plotly plot generation."""

from pathlib import Path
import polars as pl
import pandas as pd
from glucose_neuralforecast.plotting_plotly import plot_comparison_plotly

# Load data
base_path = Path("/home/antonkulaga/sources/glucose-neuralforecast")
data_path = base_path / "data/input/livia_glucose.csv"
run_path = base_path / "data/output/runs/run_20251021_153634"
predictions_dir = run_path / "predictions"

# Load original data
df = pl.read_csv(data_path)

# Filter for sequence 6
df_seq = df.filter(pl.col('unique_id') == 6)
print(f"Original data for sequence 6: {len(df_seq)} rows")
print(f"Columns: {df_seq.columns}")
print(df_seq.head())

# Load predictions
predictions = {}
for pred_file in predictions_dir.glob("predictions_*.csv"):
    model_name = pred_file.stem.replace("predictions_", "")
    pred_df = pd.read_csv(pred_file)
    pred_seq = pred_df[pred_df['unique_id'] == 6]
    
    if len(pred_seq) > 0:
        print(f"\n{model_name}: {len(pred_seq)} rows")
        print(f"Columns: {pred_seq.columns.tolist()}")
        print(pred_seq.head())
        predictions[model_name] = pred_df
        break  # Just test with one model first

# Try to create plot
print(f"\n\nAttempting to create plot with {len(predictions)} models...")
try:
    plot_comparison_plotly(
        df=df_seq,
        predictions=predictions,
        output_path=run_path,
        sequence_id=6,
        show_all_ticks=True,
        tickangle=-90,
    )
    print("✅ Plot created successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

