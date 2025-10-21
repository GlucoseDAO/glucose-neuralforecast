"""Inference module for loading trained models and making predictions."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import random
import traceback

import polars as pl
import pandas as pd
import typer
from eliot import start_action
from pycomfort.logging import to_nice_stdout
from neuralforecast import NeuralForecast

from glucose_neuralforecast.utils import resolve_base_folder
from glucose_neuralforecast.data import load_glucose_data
from glucose_neuralforecast.config import get_latest_run
from glucose_neuralforecast.plotting_plotly import plot_comparison_plotly

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utilsforecast.plotting import plot_series

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
        
        # Fix frequency string for compatibility with newer polars versions
        # Convert 'min' to 'm' (minutes) as 'min' is no longer supported
        if hasattr(nf, 'freq') and nf.freq and 'min' in nf.freq:
            nf.freq = nf.freq.replace('min', 'm')
            action.log(message_type="freq_converted", original_freq=nf.freq.replace('m', 'min'), new_freq=nf.freq)
        
        action.log(message_type="model_loaded", success=True, freq=getattr(nf, 'freq', 'unknown'))
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
            # Get all numeric columns (exclude unique_id and metric which are non-numeric)
            numeric_cols = [
                col for col in mae_metrics.columns 
                if col not in ['unique_id', 'metric'] and mae_metrics[col].dtype in [pl.Float64, pl.Float32, pl.Int32, pl.Int64]
            ]
            
            # Select unique_id and all numeric columns, then calculate mean
            mae_for_calc = mae_metrics.select(['unique_id'] + numeric_cols)
            
            # Calculate row-wise mean for numeric columns and add to dataframe
            mae_with_mean = mae_for_calc.with_columns(
                pl.concat_list(numeric_cols)
                .list.drop_nulls()
                .list.mean()
                .alias('mean_mae')
            )
            
            # Get the unique_id with the lowest mean MAE
            best_row = mae_with_mean.drop_nulls('mean_mae').sort('mean_mae').head(1)
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


def get_best_models_from_summary(
    metrics_summary_path: Path,
    top_n: int = 6,
    metric: str = 'mae'
) -> List[str]:
    """
    Get the best models based on metrics summary.
    
    Args:
        metrics_summary_path: Path to metrics_summary.csv file
        top_n: Number of best models to select
        metric: Metric to use for selection (default: 'mae', lower is better)
        
    Returns:
        List[str]: List of best model names sorted by performance
    """
    with start_action(action_type="get_best_models_from_summary", top_n=top_n, metric=metric) as action:
        if not metrics_summary_path.exists():
            action.log(message_type="metrics_summary_not_found", path=str(metrics_summary_path))
            return []
        
        summary = pl.read_csv(metrics_summary_path)
        
        # Filter to the specified metric
        metric_row = summary.filter(pl.col('metric') == metric)
        
        if len(metric_row) == 0:
            action.log(message_type="metric_not_found", metric=metric)
            return []
        
        # Get all columns with '_mean' suffix (these are the average values for each model)
        mean_cols = [col for col in metric_row.columns if col.endswith('_mean')]
        
        if len(mean_cols) == 0:
            action.log(message_type="no_mean_columns_found")
            return []
        
        # Convert to pandas for easier manipulation
        metric_df = metric_row.to_pandas()
        
        # Extract model names and their mean values
        model_scores = []
        for col in mean_cols:
            model_name = col.replace('_mean', '')
            score = metric_df[col].iloc[0]
            if pd.notna(score):
                model_scores.append((model_name, float(score)))
        
        # Sort by score (ascending for MAE - lower is better)
        model_scores.sort(key=lambda x: x[1])
        
        # Take top N models
        best_models = [name for name, score in model_scores[:top_n]]
        
        action.log(
            message_type="best_models_selected",
            models=best_models,
            scores={name: score for name, score in model_scores[:top_n]}
        )
        
        return best_models


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
    filename: str = "comparison.png",
    metrics_path: Optional[Path] = None,
    mae_threshold: Optional[float] = None,
    top_models: Optional[int] = None,
    use_plotly: bool = True
) -> bool:
    """
    Plot comparison of multiple model predictions for a specific sequence.
    Optionally filters models by MAE threshold and selects top N models by MAE.
    
    Args:
        original_df: Original data with actual values (pandas format)
        predictions: Dictionary mapping model names to their prediction DataFrames
        unique_id: The unique_id of the sequence to plot
        output_path: Directory to save the plot
        filename: Name of the output file
        metrics_path: Path to metrics CSV file for filtering models by MAE (optional)
        mae_threshold: Maximum MAE threshold to include models (None to disable filtering)
        top_models: Number of best models to show (None for all)
        use_plotly: Use plotly for interactive plots (default: True)
        
    Returns:
        bool: True if plot was successfully saved, False otherwise
    """
    with start_action(action_type="plot_model_comparison", unique_id=unique_id) as action:
        # Convert unique_id to int for comparison
        unique_id_int = int(unique_id)
        
        # Filter original data for this sequence
        df_seq = original_df[original_df['unique_id'] == unique_id_int]
        
        if len(df_seq) == 0:
            action.log(message_type="no_data_for_sequence", unique_id=unique_id)
            return False
        
        # Combine all predictions for this sequence
        model_names = list(predictions.keys())
        if len(model_names) == 0:
            action.log(message_type="no_predictions")
            return False
        
        # Filter models by MAE threshold and select top N models (only if metrics_path is provided)
        if metrics_path and metrics_path.exists() and (mae_threshold is not None or top_models is not None):
            metrics_df = pl.read_csv(metrics_path)
            mae_metrics = metrics_df.filter(pl.col('metric') == 'mae')
            
            # Get MAE for this sequence
            seq_metrics = mae_metrics.filter(pl.col('unique_id') == unique_id_int)
            
            if len(seq_metrics) > 0:
                seq_metrics_row = seq_metrics.row(0)
                seq_metrics_dict = seq_metrics.to_pandas().iloc[0].to_dict()
                
                # Create list of (model_name, mae_value) tuples
                model_mae_pairs = []
                
                for model_name in model_names:
                    # Check if model column exists and MAE is below threshold
                    if model_name in seq_metrics_dict:
                        mae_value = seq_metrics_dict[model_name]
                        # Include model if MAE is not null
                        if pd.notna(mae_value):
                            # Apply threshold filter only if specified
                            if mae_threshold is None or mae_value < mae_threshold:
                                model_mae_pairs.append((model_name, float(mae_value)))
                            elif mae_threshold is not None:
                                action.log(message_type="model_filtered_by_threshold", model=model_name, mae=float(mae_value), threshold=mae_threshold)
                        else:
                            action.log(message_type="model_mae_null", model=model_name)
                    else:
                        # If model not in metrics, include it anyway
                        action.log(message_type="model_mae_not_found", model=model_name)
                        model_mae_pairs.append((model_name, float('inf')))
                
                # Sort by MAE (ascending) and take top N
                model_mae_pairs.sort(key=lambda x: x[1])
                
                if top_models is not None and len(model_mae_pairs) > top_models:
                    action.log(message_type="selecting_top_models", total=len(model_mae_pairs), top=top_models)
                    model_mae_pairs = model_mae_pairs[:top_models]
                
                model_names = [name for name, mae in model_mae_pairs if mae != float('inf')]
                action.log(message_type="models_after_filtering", count=len(model_names), models=model_names)
        
        # Remove models with null predictions
        models_to_plot = []
        for model_name in model_names:
            pred_df = predictions.get(model_name)
            if pred_df is not None:
                # The model column name in predictions may not include the _exog suffix
                # Try both the full model name and the base name without _exog
                possible_col_names = [model_name]
                if model_name.endswith('_exog'):
                    base_name = model_name.replace('_exog', '')
                    possible_col_names.append(base_name)
                
                # Check if any of the possible column names exist with non-null values
                col_found = False
                for col_name in possible_col_names:
                    if col_name in pred_df.columns and pred_df[col_name].notna().any():
                        models_to_plot.append((model_name, col_name))
                        col_found = True
                        break
                
                if not col_found:
                    action.log(message_type="model_has_nulls", model=model_name, available_cols=list(pred_df.columns))
            else:
                action.log(message_type="model_prediction_missing", model=model_name)
        
        if len(models_to_plot) == 0:
            action.log(message_type="no_valid_models_after_filtering")
            return False
        
        # Get first model's predictions for this sequence
        first_model_name, first_col_name = models_to_plot[0]
        combined_pred = predictions[first_model_name][
            predictions[first_model_name]['unique_id'] == unique_id_int
        ].copy()
        
        # Rename the first column to match model name if needed
        if first_col_name != first_model_name and first_col_name in combined_pred.columns:
            combined_pred = combined_pred.rename(columns={first_col_name: first_model_name})
        
        # Track which models were actually added to combined predictions
        models_in_pred = [first_model_name]
        
        # Join predictions from other models
        for model_name, col_name in models_to_plot[1:]:
            model_pred = predictions[model_name][
                predictions[model_name]['unique_id'] == unique_id_int
            ].copy()
            
            if len(model_pred) > 0 and col_name in model_pred.columns:
                # Rename the column to match the model name BEFORE merging
                if col_name != model_name:
                    model_pred = model_pred.rename(columns={col_name: model_name})
                
                # Merge on ds (timestamp) and unique_id
                merge_cols = ['ds', 'unique_id']
                combined_pred = combined_pred.merge(
                    model_pred[merge_cols + [model_name]],
                    on=merge_cols,
                    how='outer'
                )
                models_in_pred.append(model_name)
        
        # Update model_names to only include those actually in combined predictions
        model_names = models_in_pred
        
        # Create plot
        output_path.mkdir(parents=True, exist_ok=True)
        
        if use_plotly:
            # Use plotly for interactive plots
            action.log(message_type="using_plotly_backend")
            
            # Convert to polars for plotly function
            df_seq_polars = pl.from_pandas(df_seq)
            
            # Create predictions dict for plotly
            predictions_for_plotly = {}
            for model_name in model_names:
                # Check if model column exists in combined_pred
                if model_name in combined_pred.columns:
                    model_pred = combined_pred[['unique_id', 'ds', model_name]].copy()
                    predictions_for_plotly[model_name] = model_pred
            
            # Use plotly comparison plot
            success = plot_comparison_plotly(
                df=df_seq_polars,
                predictions=predictions_for_plotly,
                output_path=output_path / 'plots',  # plot_comparison_plotly will add 'comparison' subdir
                sequence_id=unique_id_int,
                show_all_ticks=True,
                tickangle=-90,
            )
            
            action.log(message_type="plot_saved_plotly", models_plotted=len(model_names), success=success)
            return success
        else:
            # Use matplotlib for static plots
            action.log(message_type="using_matplotlib_backend")
            
            # Convert unique_id to string for plot_series
            df_seq_plot = df_seq.copy()
            df_seq_plot['unique_id'] = df_seq_plot['unique_id'].astype(str)
            
            combined_pred_plot = combined_pred.copy()
            combined_pred_plot['unique_id'] = combined_pred_plot['unique_id'].astype(str)
            
            fig = plot_series(
                df_seq_plot,
                combined_pred_plot,
                models=model_names,
                level=None,
                max_insample_length=None,  # Show all historical points
                plot_random=False,
                ids=[str(unique_id_int)],
                engine="matplotlib"
            )

            # Set larger figure size for better visualization and more compact layout
            fig.set_size_inches(18, 8)

            # Make ground truth (y) line bolder for better visibility and rename label
            for ax in fig.get_axes():
                for line in ax.get_lines():
                    label = line.get_label()
                    if label == 'y':  # Ground truth line
                        line.set_label('Ground Truth Glucose')
                        line.set_linewidth(3)
                        line.set_zorder(10)  # Bring to front
                    else:
                        line.set_linewidth(1.5)

                # Update legend with new label
                ax.legend(loc='best')

                # Improve x-axis formatting
                # Rotate labels and ensure proper alignment
                for label in ax.get_xticklabels():
                    label.set_rotation(90)
                    label.set_horizontalalignment('center')
                    label.set_verticalalignment('top')
                    label.set_fontsize(8)
                
                # Configure tick parameters
                ax.tick_params(axis='x', which='major', length=6, width=1, pad=2)
                
                # Add grid for better readability
                ax.grid(True, which='major', alpha=0.3, linestyle='-', axis='both')

            # Save plot
            plot_file = output_path / filename
            fig.savefig(str(plot_file), dpi=150, bbox_inches='tight')
            action.log(message_type="plot_saved", file=str(plot_file), models_plotted=len(model_names))
            return True


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
        help="Specific training run ID to use for inference. If not provided, automatically uses the most recent training run"
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
    ),
    mae_threshold: float = typer.Option(
        40.0,
        "--mae-threshold",
        help="Maximum MAE threshold to include models in plot (default: 40.0)"
    ),
    top_models: Optional[int] = typer.Option(
        5,
        "--top-models",
        help="Number of best models to show in plot (default: 5, use 0 or None for all)"
    ),
    plotly: bool = typer.Option(
        True,
        "--plotly/--no-plotly",
        help="Use plotly for interactive plots (default: True). Use --no-plotly for matplotlib."
    )
) -> None:
    """
    Run inference with trained models on glucose data.
    
    Can predict on all data or cherry-pick a specific sequence based on metrics.
    Generates comparison plots showing predictions from all selected models.
    Filters models by MAE threshold and shows only top N models by MAE.
    """
    with start_action(action_type="inference_command") as main_action:
        # Set up eliot logging to stdout
        to_nice_stdout()
        
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
            # Use specified run_id
            output_path = base_output_path / 'runs' / run_id
            if not output_path.exists():
                typer.echo(f"❌ Training run not found: {run_id}")
                typer.echo(f"   Expected path: {output_path}")
                typer.echo(f"   Use 'list-runs' to see available training runs")
                raise typer.Exit(1)
            typer.echo(f"🎯 Using training run: {run_id}")
        else:
            # Auto-detect latest run
            latest_run = get_latest_run(base_output_path)
            if latest_run:
                run_id = latest_run
                output_path = base_output_path / 'runs' / run_id
                typer.echo(f"🎯 Auto-detected latest training run: {run_id}")
            else:
                # Legacy mode: use base output directory directly
                output_path = base_output_path
                typer.echo(f"⚠️  No training runs found. Using legacy mode.")
                typer.echo(f"   Run 'train' to create a new training run.")
        
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
            # Auto-select best models from metrics_summary if available
            metrics_summary_path = output_path / 'metrics_summary.csv'
            if metrics_summary_path.exists():
                typer.echo(f"\n🏆 Auto-selecting best models from metrics summary...")
                best_models = get_best_models_from_summary(metrics_summary_path, top_n=6)
                
                # Filter to only those that are actually available
                models_to_use = [m for m in best_models if m in available_models]
                
                if len(models_to_use) == 0:
                    typer.echo(f"⚠️  No best models found in available models, using all available models")
                    models_to_use = available_models
                else:
                    typer.echo(f"   Selected top {len(models_to_use)} models by MAE")
                    for i, model_name in enumerate(models_to_use, 1):
                        typer.echo(f"   {i}. {model_name}")
            else:
                typer.echo(f"\n⚠️  Metrics summary not found, using all available models")
                models_to_use = available_models
        
        typer.echo(f"\n🔧 Using models ({len(models_to_use)}): {', '.join(models_to_use)}")
        main_action.log(message_type="models_selected", models=models_to_use)
        
        # Auto-enable cherry-pick if plot is enabled and no sequence selection is specified
        effective_cherry_pick = cherry_pick
        if plot and not unique_id and not cherry_pick:
            effective_cherry_pick = True
            typer.echo(f"\n🍒 Plot generation enabled - auto-enabling cherry-pick mode (best sequence)")
        
        # Determine which sequence to predict
        selected_unique_id = None
        
        if unique_id:
            selected_unique_id = unique_id
            typer.echo(f"\n🎯 Using specified unique_id: {selected_unique_id}")
        elif effective_cherry_pick:
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
        
        # Determine if we need exogenous variables based on models
        needs_exogenous = any('_exog' in model_name for model_name in models_to_use)
        
        if needs_exogenous:
            typer.echo(f"\n🔗 Detected models requiring exogenous variables")
        
        # Load data
        typer.echo(f"\n📊 Loading data from: {input_path}")
        typer.echo(f"   Include exogenous variables: {needs_exogenous}")
        df = load_glucose_data(input_path, include_exogenous=needs_exogenous)
        
        # Store full dataframe for plotting later
        df_full_for_plot = None
        
        # Filter to selected sequence if specified
        if selected_unique_id is not None:
            # Convert string unique_id to int for comparison
            unique_id_int = int(selected_unique_id)
            df_filtered = df.filter(pl.col('unique_id') == unique_id_int)
            if len(df_filtered) == 0:
                typer.echo(f"❌ No data found for unique_id: {selected_unique_id}")
                raise typer.Exit(1)
            
            # Load config to get horizon and input_size
            config_path = output_path / 'config.yaml'
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                horizon = config.get('horizon', 12)
                input_size = config.get('input_size', 48)
            else:
                # Default values
                horizon = 12
                input_size = 48
            
            # For plotting: we want to show last 48 timepoints with predictions for last 12
            # For inference: provide data up to -12, model predicts next 12 points
            # Total data needed: at least 48 points for plotting
            plot_points = 48
            
            if len(df_filtered) < plot_points:
                typer.echo(f"⚠️  Sequence {selected_unique_id} has only {len(df_filtered)} rows, need at least {plot_points}")
                typer.echo(f"   Using all available data for this sequence")
                # Use all data for inference (will predict beyond it)
                df = df_filtered
                df_full_for_plot = df_filtered
            else:
                # Take last 48 points for plotting (these include the last 12 we want to predict)
                df_full_for_plot = df_filtered.tail(plot_points)
                
                # For inference: provide everything except the last 12 points
                # This way predictions will cover the last 12 points
                df_for_inference = df_filtered.head(len(df_filtered) - horizon)
                
                if len(df_for_inference) < input_size:
                    typer.echo(f"⚠️  Not enough data for inference (need at least {input_size} points)")
                    typer.echo(f"   Using all available data")
                    df = df_filtered
                else:
                    df = df_for_inference
                    typer.echo(f"   Using last {plot_points} points for plotting (36 actual + 12 predicted)")
                    typer.echo(f"   Inference on {len(df)} points (excluding last {horizon} to predict them)")
        
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
        if plot:
            if selected_unique_id is not None:
                typer.echo(f"\n📈 Generating comparison plot...")
                
                # Use df_full_for_plot if available (contains last 48 points), otherwise use df
                df_for_plot = df_full_for_plot if df_full_for_plot is not None else df
                
                # Convert to pandas for plotting
                df_pandas = df_for_plot.to_pandas() if isinstance(df_for_plot, pl.DataFrame) else df_for_plot
                
                metrics_path = output_path / 'metrics.csv'
                # Handle top_models=0 as None (show all models)
                effective_top_models = top_models if top_models and top_models > 0 else None
                
                # Don't use mae_threshold since we already pre-selected the best models
                # Pass None to disable threshold filtering
                try:
                    plot_success = plot_model_comparison(
                        df_pandas,
                        predictions,
                        selected_unique_id,
                        output_path,  # plot_model_comparison will create plots/comparison subdirs
                        filename=f'comparison_{selected_unique_id}.png',
                        metrics_path=metrics_path if metrics_path.exists() else None,
                        mae_threshold=None,  # Disable threshold filtering since models are pre-selected
                        top_models=None,  # Don't further filter since models are already pre-selected
                        use_plotly=plotly
                    )
                except Exception as e:
                    typer.echo(f"   ❌ Error during plot generation: {e}")
                    import traceback
                    traceback.print_exc()
                    plot_success = False
                    
                if plot_success:
                    if plotly:
                        # Verify files actually exist before claiming success
                        plots_dir = output_path / 'plots' / 'comparison'
                        html_path = plots_dir / f'comparison_{selected_unique_id}.html'
                        png_path = plots_dir / f'comparison_{selected_unique_id}.png'
                        if html_path.exists():
                            typer.echo(f"   ✅ Interactive plot saved to: {html_path}")
                        else:
                            typer.echo(f"   ⚠️  HTML plot not found at: {html_path}")
                        if png_path.exists():
                            typer.echo(f"   ✅ Static plot saved to: {png_path}")
                        else:
                            typer.echo(f"   ⚠️  PNG plot not found at: {png_path}")
                    else:
                        plots_dir = output_path / 'plots' / 'comparison'
                        typer.echo(f"   ✅ Plot saved to: {plots_dir / f'comparison_{selected_unique_id}.png'}")
                else:
                    typer.echo(f"   ❌ Failed to generate plot for {selected_unique_id}")
            else:
                typer.echo(f"\n⚠️  Plot generation requires a specific sequence (use --cherry-pick or --unique-id)")
        
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

