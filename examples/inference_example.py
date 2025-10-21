#!/usr/bin/env python3
"""
Example script demonstrating the inference module functionality.

This script shows how to:
1. Load trained models
2. Run inference with cherry-pick mode
3. Compare multiple models
4. Generate comparison plots
"""

from pathlib import Path
from glucose_neuralforecast.utils import resolve_base_folder
from glucose_neuralforecast.data import load_glucose_data
from glucose_neuralforecast.inference import (
    get_available_trained_models,
    cherry_pick_sequence,
    run_inference,
    plot_model_comparison,
)


def main() -> None:
    """Run inference example."""
    print("🔮 Glucose NeuralForecast - Inference Example\n")
    
    # Setup paths
    base = resolve_base_folder()
    data_dir = base / 'data'
    input_path = data_dir / 'input' / 'livia_glucose.csv'
    output_path = data_dir / 'output'
    models_dir = output_path / 'models'
    metrics_path = output_path / 'metrics.csv'
    
    print(f"Base folder: {base}")
    print(f"Models directory: {models_dir}")
    
    # Check if models exist
    if not models_dir.exists():
        print("\n❌ Models directory not found. Please train models first:")
        print("   glucose-train train")
        return
    
    # Get available trained models
    available_models = get_available_trained_models(models_dir)
    
    if len(available_models) == 0:
        print("\n❌ No trained models found. Please train models first:")
        print("   glucose-train train")
        return
    
    print(f"\n📊 Found {len(available_models)} trained models:")
    for model in available_models:
        print(f"  • {model}")
    
    # Cherry-pick best sequence
    if not metrics_path.exists():
        print("\n❌ Metrics file not found. Please ensure models were trained with metrics.")
        return
    
    print("\n🍒 Cherry-picking best sequence...")
    selected_id = cherry_pick_sequence(metrics_path, best=True)
    print(f"   Selected sequence: {selected_id}")
    
    # Select a few models for demonstration (or use all)
    # For demo purposes, let's use just 3-5 models if more are available
    models_to_use = available_models[:min(5, len(available_models))]
    print(f"\n🔧 Using models for comparison: {', '.join(models_to_use)}")
    
    # Determine if we need exogenous variables based on models
    needs_exogenous = any('_exog' in model_name for model_name in models_to_use)
    if needs_exogenous:
        print("   🔗 Detected models requiring exogenous variables")
    
    # Load data
    print(f"\n📊 Loading data (exogenous: {needs_exogenous})...")
    df = load_glucose_data(input_path, include_exogenous=needs_exogenous)
    
    # Filter to selected sequence
    df_filtered = df.filter(df['unique_id'] == selected_id)
    print(f"   Filtered to sequence {selected_id}: {len(df_filtered)} rows")
    
    # Run inference
    print("\n🔮 Running inference...")
    predictions = run_inference(df_filtered, models_to_use, models_dir)
    
    if len(predictions) == 0:
        print("❌ No successful predictions")
        return
    
    print(f"✅ Successfully generated predictions from {len(predictions)} models")
    
    # Print prediction summary
    for model_name, pred_df in predictions.items():
        print(f"   • {model_name}: {len(pred_df)} predictions")
    
    # Generate comparison plot
    print("\n📈 Generating comparison plot...")
    plots_dir = output_path / 'plots' / 'comparison'
    
    df_pandas = df_filtered.to_pandas()
    
    plot_model_comparison(
        df_pandas,
        predictions,
        selected_id,
        plots_dir,
        filename=f'example_comparison_{selected_id}.png'
    )
    
    plot_path = plots_dir / f'example_comparison_{selected_id}.png'
    if plot_path.exists():
        print(f"✅ Plot saved to: {plot_path}")
    else:
        print("⚠️  Plot not generated (plotting library may not be available)")
    
    print("\n✨ Example complete!")
    print("\n💡 Try running with different options:")
    print("   • Random sequence: cherry_pick_sequence(metrics_path, best=False)")
    print("   • All models: models_to_use = available_models")
    print("   • Different sequence: df.filter(df['unique_id'] == 'your_sequence_id')")


if __name__ == "__main__":
    main()

