"""Inference module for loading trained models and making predictions."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import random

import polars as pl
import pandas as pd
import typer
from eliot import start_action
from neuralforecast import NeuralForecast

from glucose_neuralforecast.utils import resolve_base_folder
from glucose_neuralforecast.data import load_glucose_data

# Optional plotting
HAS_PLOTTING = False
try:
    from utilsforecast.plotting import plot_series
    HAS_PLOTTING = True
except ImportError:
    pass

app = typer.Typer()


def load_trained_model(model_path: Path) -> NeuralForecast:
    """
    Load a trained NeuralForecast model from disk.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        NeuralForecast: Loaded model ready for inference
    """
    with start_action(action_type="load_trained_model", path=str(model_path)) as action:
        nf = NeuralForecast.load(path=str(model_path))
        action.log(message_type="model_loaded", success=True)
        return nf


def get_available_trained_models(models_dir: Path) -> List[str]:
    """
    Get list of available trained models in the models directory.
    
    Args:
        models_dir: Path to the directory containing trained models
        
    Returns:
        List[str]: List of model names that have been trained
    """
    if not models_dir.exists():
        return []
    
    model_names = []
    for model_path in models_dir.iterdir():
        if model_path.is_dir():
            # Check if it has the required files
            if (model_path / 'configuration.pkl').exists():
                model_names.append(model_path.name)
    
    return sorted(model_names)


def cherry_pick_sequence(
    metrics_path: Path,
    best: bool = True,
    seed: Optional[int] = None
) -> str:
    """
    Cherry-pick a sequence from metrics based on best MAE or randomly.
    
    Args:
        metrics_path: Path to metrics.csv file
        best: If True, pick sequence with best (lowest) MAE. If False, pick randomly.
        seed: Random seed for reproducibility when best=False
        
    Returns:
        str: The unique_id of the selected sequence
    """
    with start_action(action_type="cherry_pick_sequence", best=best, seed=seed) as action:
        metrics = pl.read_csv(metrics_path)
        
        # Filter to only MAE metric
        mae_metrics = metrics.filter(pl.col('metric') == 'mae')
        
        if len(mae_metrics) == 0:
            raise ValueError("No MAE metrics found in metrics file")
        
        unique_ids = mae_metrics['unique_id'].to_list()
        
        if best:
            # Find sequence with best average MAE across all models
            # Get all model columns (exclude unique_id and metric)
            model_cols = [col for col in mae_metrics.columns if col not in ['unique_id', 'metric']]
            
            # Calculate mean MAE across all models for each unique_id
            mae_with_mean = mae_metrics.with_columns([
                pl.mean_horizontal([pl.col(col) for col in model_cols]).alias('mean_mae')
            ])
            
            # Get the unique_id with the lowest mean MAE
            best_row = mae_with_mean.sort('mean_mae').head(1)
            selected_id = str(best_row['unique_id'][0])
            best_mae = float(best_row['mean_mae'][0])
            
            action.log(message_type="best_sequence_selected", unique_id=selected_id, mean_mae=best_mae)
        else:
            # Pick randomly
            if seed is not None:
                random.seed(seed)
            selected_id = str(random.choice(unique_ids))
            action.log(message_type="random_sequence_selected", unique_id=selected_id)
        
        return selected_id


def run_inference(
    data_df: pl.DataFrame,
    models_to_use: List[str],
    models_dir: Path
) -> Dict[str, pd.DataFrame]:
    """
    Run inference with multiple models on the provided data.
    
    Args:
        data_df: DataFrame with glucose data (polars format)
        models_to_use: List of model names to use for inference
        models_dir: Path to directory containing trained models
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping model names to their predictions
    """
    with start_action(action_type="run_inference", models=models_to_use) as action:
        predictions = {}
        
        for model_name in models_to_use:
            model_path = models_dir / model_name
            
            if not model_path.exists():
                action.log(message_type="model_not_found", model=model_name)
                continue
            
            try:
                with start_action(action_type="predict_with_model", model=model_name):
                    # Load the trained model
                    nf = load_trained_model(model_path)
                    
                    # Run prediction
                    pred_df = nf.predict(df=data_df)
                    
                    # Convert to pandas if needed
                    if isinstance(pred_df, pl.DataFrame):
                        pred_df = pred_df.to_pandas()
                    
                    predictions[model_name] = pred_df
                    action.log(message_type="prediction_completed", model=model_name, shape=pred_df.shape)
                    
            except Exception as e:
                action.log(message_type="prediction_error", model=model_name, error=str(e))
                continue
        
        action.log(message_type="inference_completed", successful_models=len(predictions))
        return predictions


def plot_model_comparison(
    original_df: pd.DataFrame,
    predictions: Dict[str, pd.DataFrame],
    unique_id: str,
    output_path: Path,
    filename: str = "comparison.png"
) -> None:
    """
    Plot comparison of multiple model predictions for a specific sequence.
    
    Args:
        original_df: Original data with actual values (pandas format)
        predictions: Dictionary mapping model names to their prediction DataFrames
        unique_id: The unique_id of the sequence to plot
        output_path: Directory to save the plot
        filename: Name of the output file
    """
    if not HAS_PLOTTING:
        print("Plotting library not available. Skipping plot generation.")
        return
    
    with start_action(action_type="plot_model_comparison", unique_id=unique_id) as action:
        try:
            # Filter original data for this sequence
            df_seq = original_df[original_df['unique_id'] == unique_id]
            
            if len(df_seq) == 0:
                action.log(message_type="no_data_for_sequence", unique_id=unique_id)
                return
            
            # Combine all predictions for this sequence
            # Start with the first prediction
            model_names = list(predictions.keys())
            if len(model_names) == 0:
                action.log(message_type="no_predictions")
                return
            
            # Get first model's predictions for this sequence
            combined_pred = predictions[model_names[0]][
                predictions[model_names[0]]['unique_id'] == unique_id
            ].copy()
            
            # Rename the prediction column to include model name
            if model_names[0] in combined_pred.columns:
                combined_pred = combined_pred.rename(columns={model_names[0]: model_names[0]})
            
            # Join predictions from other models
            for model_name in model_names[1:]:
                model_pred = predictions[model_name][
                    predictions[model_name]['unique_id'] == unique_id
                ].copy()
                
                if len(model_pred) > 0:
                    # Merge on ds (timestamp) and unique_id
                    merge_cols = ['ds', 'unique_id']
                    if model_name in model_pred.columns:
                        combined_pred = combined_pred.merge(
                            model_pred[merge_cols + [model_name]],
                            on=merge_cols,
                            how='outer'
                        )
            
            # Create plot
            output_path.mkdir(parents=True, exist_ok=True)
            
            fig = plot_series(
                df_seq,
                combined_pred,
                models=model_names,
                level=None,
                max_insample_length=500,
                plot_random=False,
                ids=[unique_id]
            )
            
            # Save plot
            plot_file = output_path / filename
            fig.write_image(str(plot_file))
            action.log(message_type="plot_saved", file=str(plot_file))
            
        except Exception as e:
            action.log(message_type="plotting_error", error=str(e))


@app.command()
def predict(
    data_file: Optional[str] = typer.Option(
        None,
        "--data-file",
        "-d",
        help="Path to the glucose CSV file. If not provided, uses data/input/livia_glucose.csv"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Base output directory. If not provided, uses data/output"
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Specific training run ID to use for inference. If not provided, uses data/output directly (legacy mode)"
    ),
    models: Optional[str] = typer.Option(
        None,
        "--models",
        "-m",
        help="Comma-separated list of models to use (e.g., 'NBEATS,NHITS,LSTM'). If not provided, uses all available trained models"
    ),
    cherry_pick: bool = typer.Option(
        False,
        "--cherry-pick",
        "-c",
        help="Use cherry-pick mode to select a specific sequence"
    ),
    best: bool = typer.Option(
        True,
        "--best/--random",
        help="In cherry-pick mode, select sequence with best MAE (--best) or random (--random)"
    ),
    unique_id: Optional[str] = typer.Option(
        None,
        "--unique-id",
        "-u",
        help="Specific unique_id to predict. Overrides cherry-pick mode."
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed for reproducibility when using random selection"
    ),
    save_predictions: bool = typer.Option(
        True,
        "--save-predictions/--no-save-predictions",
        help="Save predictions to CSV files"
    ),
    plot: bool = typer.Option(
        True,
        "--plot/--no-plot",
        help="Generate comparison plot"
    )
) -> None:
    """
    Run inference with trained models on glucose data.
    
    Can predict on all data or cherry-pick a specific sequence based on metrics.
    Generates comparison plots showing predictions from all selected models.
    """
    with start_action(action_type="inference_command") as main_action:
        # Resolve base folder
        base = resolve_base_folder()
        main_action.log(message_type="base_folder", path=str(base))
        typer.echo(f"Base folder: {base}")
        
        # Set up paths
        data_dir = base / 'data'
        input_path = Path(data_file) if data_file else data_dir / 'input' / 'livia_glucose.csv'
        base_output_path = Path(output_dir) if output_dir else data_dir / 'output'
        
        # Determine actual output path based on run_id
        if run_id:
            output_path = base_output_path / 'runs' / run_id
            if not output_path.exists():
                typer.echo(f"❌ Training run not found: {run_id}")
                typer.echo(f"   Expected path: {output_path}")
                typer.echo(f"   Use 'list-runs' to see available training runs")
                raise typer.Exit(1)
            typer.echo(f"🎯 Using training run: {run_id}")
        else:
            # Legacy mode: use base output directory directly
            output_path = base_output_path
            typer.echo(f"⚠️  Using legacy mode (no run_id specified)")
        
        models_dir = output_path / 'models'
        
        if not models_dir.exists():
            typer.echo(f"❌ Models directory not found: {models_dir}")
            typer.echo(f"   Please train models first using: train")
            if not run_id:
                typer.echo(f"   Or specify a training run with: --run-id <run_id>")
            raise typer.Exit(1)
        
        # Get available models
        available_models = get_available_trained_models(models_dir)
        
        if len(available_models) == 0:
            typer.echo(f"❌ No trained models found in: {models_dir}")
            typer.echo(f"   Please train models first using: train")
            raise typer.Exit(1)
        
        typer.echo(f"\n📊 Available trained models ({len(available_models)}):")
        for model_name in available_models:
            typer.echo(f"  • {model_name}")
        
        # Select models to use
        if models:
            models_to_use = [name.strip() for name in models.split(',')]
            # Validate that all requested models are available
            missing_models = [m for m in models_to_use if m not in available_models]
            if missing_models:
                typer.echo(f"❌ Requested models not found: {', '.join(missing_models)}")
                typer.echo(f"   Available models: {', '.join(available_models)}")
                raise typer.Exit(1)
        else:
            models_to_use = available_models
        
        typer.echo(f"\n🔧 Using models ({len(models_to_use)}): {', '.join(models_to_use)}")
        main_action.log(message_type="models_selected", models=models_to_use)
        
        # Determine which sequence to predict
        selected_unique_id = None
        
        if unique_id:
            selected_unique_id = unique_id
            typer.echo(f"\n🎯 Using specified unique_id: {selected_unique_id}")
        elif cherry_pick:
            metrics_path = output_path / 'metrics.csv'
            if not metrics_path.exists():
                typer.echo(f"❌ Metrics file not found: {metrics_path}")
                typer.echo(f"   Cherry-pick mode requires metrics from training")
                raise typer.Exit(1)
            
            mode = "best MAE" if best else "random"
            typer.echo(f"\n🍒 Cherry-picking sequence ({mode})...")
            selected_unique_id = cherry_pick_sequence(metrics_path, best=best, seed=seed)
            typer.echo(f"   Selected unique_id: {selected_unique_id}")
        
        main_action.log(message_type="sequence_selection", unique_id=selected_unique_id)
        
        # Load data
        typer.echo(f"\n📊 Loading data from: {input_path}")
        df = load_glucose_data(input_path, include_exogenous=False)
        
        # Filter to selected sequence if specified
        if selected_unique_id is not None:
            # Convert string unique_id to int for comparison
            unique_id_int = int(selected_unique_id)
            df_filtered = df.filter(pl.col('unique_id') == unique_id_int)
            if len(df_filtered) == 0:
                typer.echo(f"❌ No data found for unique_id: {selected_unique_id}")
                raise typer.Exit(1)
            df = df_filtered
            typer.echo(f"   Filtered to sequence {selected_unique_id}: {len(df)} rows")
        
        main_action.log(
            message_type="data_loaded",
            shape=df.shape,
            unique_sequences=df['unique_id'].n_unique()
        )
        
        # Run inference
        typer.echo(f"\n🔮 Running inference with {len(models_to_use)} models...")
        predictions = run_inference(df, models_to_use, models_dir)
        
        if len(predictions) == 0:
            typer.echo(f"❌ No successful predictions")
            raise typer.Exit(1)
        
        typer.echo(f"✅ Successfully generated predictions from {len(predictions)} models")
        
        # Save predictions
        if save_predictions:
            predictions_dir = output_path / 'predictions'
            predictions_dir.mkdir(parents=True, exist_ok=True)
            
            typer.echo(f"\n💾 Saving predictions...")
            for model_name, pred_df in predictions.items():
                pred_path = predictions_dir / f'predictions_{model_name}.csv'
                if isinstance(pred_df, pl.DataFrame):
                    pred_df.write_csv(pred_path)
                else:
                    pl.from_pandas(pred_df).write_csv(pred_path)
                typer.echo(f"   • {model_name}: {pred_path}")
            
            main_action.log(message_type="predictions_saved", directory=str(predictions_dir))
        
        # Generate comparison plot
        if plot and HAS_PLOTTING:
            if selected_unique_id is not None:
                typer.echo(f"\n📈 Generating comparison plot...")
                plots_dir = output_path / 'plots' / 'comparison'
                
                # Convert df to pandas for plotting
                df_pandas = df.to_pandas() if isinstance(df, pl.DataFrame) else df
                
                plot_model_comparison(
                    df_pandas,
                    predictions,
                    selected_unique_id,
                    plots_dir,
                    filename=f'comparison_{selected_unique_id}.png'
                )
                typer.echo(f"   ✅ Plot saved to: {plots_dir / f'comparison_{selected_unique_id}.png'}")
            else:
                typer.echo(f"\n⚠️  Plot generation requires a specific sequence (use --cherry-pick or --unique-id)")
        elif plot and not HAS_PLOTTING:
            typer.echo(f"\n⚠️  Plotting library not available. Install with: pip install kaleido")
        
        typer.echo(f"\n✨ Inference complete!")


@app.command()
def list_trained_models(
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory with trained models. If not provided, uses data/output"
    )
) -> None:
    """
    List all available trained models.
    """
    base = resolve_base_folder()
    data_dir = base / 'data'
    output_path = Path(output_dir) if output_dir else data_dir / 'output'
    models_dir = output_path / 'models'
    
    if not models_dir.exists():
        typer.echo(f"❌ Models directory not found: {models_dir}")
        return
    
    available_models = get_available_trained_models(models_dir)
    
    if len(available_models) == 0:
        typer.echo(f"❌ No trained models found in: {models_dir}")
        return
    
    typer.echo(f"\n📊 Available trained models ({len(available_models)}):\n")
    for model_name in available_models:
        model_path = models_dir / model_name
        typer.echo(f"  • {model_name}")
        typer.echo(f"    Path: {model_path}")
    
    typer.echo(f"\n💡 Use these models with: predict --models \"MODEL1,MODEL2,...\"\n")


if __name__ == "__main__":
    app()

