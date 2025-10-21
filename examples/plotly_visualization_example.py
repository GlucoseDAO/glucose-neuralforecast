"""Example script demonstrating plotly-based visualization for glucose forecasting."""

from pathlib import Path
import polars as pl
import pandas as pd

from glucose_neuralforecast.plotting_plotly import (
    plot_predictions_plotly,
    plot_comparison_plotly,
    create_interactive_dashboard,
)
from glucose_neuralforecast.data import load_glucose_data
from glucose_neuralforecast.inference import load_trained_model
from glucose_neuralforecast.utils import resolve_base_folder


def main() -> None:
    """Run plotly visualization examples."""
    
    # Set up paths
    base = resolve_base_folder()
    data_path = base / 'data' / 'input' / 'livia_glucose.csv'
    
    # Example run directory - update this to your actual run
    run_dir = base / 'data' / 'output' / 'runs' / 'run_20251021_153634'
    
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        print("Please update the script with your actual run directory")
        return
    
    print("📊 Loading data...")
    df = load_glucose_data(data_path, include_exogenous=False)
    
    # Example 1: Plot predictions for a single model using plotly
    print("\n📈 Example 1: Plotting predictions with plotly")
    
    models_dir = run_dir / 'models'
    
    # Find a model directory (e.g., 'LSTM' or 'NHITS')
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if model_dirs:
        model_dir = model_dirs[0]
        model_name = model_dir.name
        print(f"   Using model: {model_name}")
        
        # Load model and make predictions
        nf = load_trained_model(model_dir)
        predictions = nf.predict(df=df)
        
        # Convert predictions to pandas
        predictions_df = predictions.to_pandas() if isinstance(predictions, pl.DataFrame) else predictions
        
        # Plot with plotly
        output_path = run_dir / 'plotly_examples'
        output_path.mkdir(exist_ok=True)
        
        print(f"   Saving plots to: {output_path}")
        plot_predictions_plotly(
            df=df,
            cv_df=predictions_df,
            model_name=model_name,
            output_path=output_path,
            max_sequences=3,
            show_all_ticks=True,  # Show all time point ticks
            tickangle=-90,        # Vertical labels
            height=600,
            width=1400,
        )
        print(f"   ✅ Plots saved as both HTML (interactive) and PNG (static)")
    
    # Example 2: Compare multiple models
    print("\n📊 Example 2: Model comparison with plotly")
    
    # Load predictions from multiple models
    predictions_dict = {}
    for model_dir in model_dirs[:3]:  # Compare first 3 models
        try:
            model_name = model_dir.name
            print(f"   Loading {model_name}...")
            nf = load_trained_model(model_dir)
            predictions = nf.predict(df=df)
            predictions_df = predictions.to_pandas() if isinstance(predictions, pl.DataFrame) else predictions
            predictions_dict[model_name] = predictions_df
        except Exception as e:
            print(f"   ⚠️  Skipped {model_name}: {e}")
    
    if len(predictions_dict) > 1:
        # Get a sequence to plot
        first_predictions = next(iter(predictions_dict.values()))
        sequence_id = first_predictions['unique_id'].iloc[0]
        
        print(f"   Comparing models for sequence: {sequence_id}")
        plot_comparison_plotly(
            df=df,
            predictions=predictions_dict,
            output_path=output_path,
            sequence_id=sequence_id,
            show_all_ticks=True,
            tickangle=-90,
            height=800,
            width=1600,
        )
        print(f"   ✅ Comparison plot saved")
    
    # Example 3: Create interactive dashboard
    print("\n🎛️  Example 3: Creating interactive dashboard")
    
    if len(predictions_dict) > 1:
        # Load metrics for context
        metrics_path = run_dir / 'metrics.csv'
        metrics_df = None
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
        
        create_interactive_dashboard(
            df=df,
            predictions=predictions_dict,
            metrics=metrics_df,
            output_path=output_path,
            max_sequences=5,
        )
        print(f"   ✅ Dashboard saved to: {output_path / 'plots' / 'dashboard' / 'dashboard.html'}")
    
    print("\n✨ Done! Check the output directory for interactive HTML and static PNG files.")
    print(f"   Output directory: {output_path}")
    print("\nKey features of plotly plots:")
    print("  • Interactive zooming and panning")
    print("  • Hover tooltips with exact values")
    print("  • All time points visible with vertical labels")
    print("  • Both HTML (interactive) and PNG (static) formats")


if __name__ == "__main__":
    main()

