from pathlib import Path
from typing import Optional, List, Dict, Any
import traceback

import polars as pl
import pandas as pd
import typer
from eliot import start_action
from pycomfort.logging import to_nice_file
from neuralforecast import NeuralForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mse, rmse, mape

from glucose_neuralforecast.utils import resolve_base_folder
from glucose_neuralforecast.data import load_glucose_data
from glucose_neuralforecast.plotting import plot_predictions
from glucose_neuralforecast.models import (
    get_model_list,
    get_available_models,
    get_default_models,
    get_models_by_category
)

app = typer.Typer()


@app.command()
def list_models() -> None:
    """
    List all available models that can be trained.
    """
    typer.echo("\n📊 Available NeuralForecast Models:\n")
    
    models_by_category = get_models_by_category()
    default_models = get_default_models()
    
    for category, models in models_by_category.items():
        typer.echo(f"\n{category}:")
        for model in models:
            typer.echo(f"  • {model}")
    
    typer.echo(f"\n\n📝 Total: {sum(len(models) for models in models_by_category.values())} models available")
    typer.echo("\nUsage: glucose-train --models \"NBEATS,NHITS,LSTM\"")
    typer.echo(f"\nDefault models (if --models not specified): {len(default_models)} models")
    typer.echo("  NBEATS, NHITS, NBEATSx")
    typer.echo("  LSTM, GRU, DilatedRNN")
    typer.echo("  TCN, BiTCN")
    typer.echo("  DLinear, NLinear")
    typer.echo("  TiDE, MLP\n")


@app.command()
def train(
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
        help="Directory to save model outputs. If not provided, uses data/output"
    ),
    horizon: int = typer.Option(
        12,
        "--horizon",
        "-h",
        help="Forecast horizon (number of time steps to predict)"
    ),
    input_size: int = typer.Option(
        48,
        "--input-size",
        "-i",
        help="Input size for the model (number of historical time steps)"
    ),
    max_steps: int = typer.Option(
        1000,
        "--max-steps",
        "-s",
        help="Maximum training steps for each model"
    ),
    models_to_train: Optional[str] = typer.Option(
        None,
        "--models",
        "-m",
        help="Comma-separated list of models to train (e.g., 'NBEATS,NHITS,LSTM'). If not provided, trains: NBEATS,NHITS,LSTM,DLinear"
    ),
    n_windows: int = typer.Option(
        3,
        "--n-windows",
        "-n",
        help="Number of cross-validation windows for evaluation"
    ),
    test_size: Optional[int] = typer.Option(
        None,
        "--test-size",
        "-t",
        help="Size of test set. If provided, n_windows is ignored."
    ),
    log_file: Optional[str] = typer.Option(
        None,
        "--log-file",
        "-l",
        help="Path to log file. If not provided, uses data/output/training.log"
    )
) -> None:
    """
    Train multiple NeuralForecast models on glucose data with cross-validation.
    Models are saved individually and metrics (MAE, MSE, RMSE, MAPE) are calculated and saved as CSV.
    """
    with start_action(action_type="train_neuralforecast") as main_action:
        # Resolve base folder
        base = resolve_base_folder()
        main_action.log(message_type="base_folder", path=str(base))
        typer.echo(f"Base folder: {base}")
        
        # Set up paths
        data_dir = base / 'data'
        input_path = Path(data_file) if data_file else data_dir / 'input' / 'livia_glucose.csv'
        output_path = Path(output_dir) if output_dir else data_dir / 'output'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create models directory
        models_path = output_path / 'models'
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        logs_dir = base / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging with pycomfort
        log_basename = 'training'
        to_nice_file(
            json_path=str(logs_dir / f'{log_basename}.json'),
            log_path=str(logs_dir / f'{log_basename}.log')
        )
        
        main_action.log(message_type="logging_setup", logs_dir=str(logs_dir))
        typer.echo(f"Logs directory: {logs_dir}")
        
        main_action.log(message_type="loading_data", path=str(input_path))
        typer.echo(f"Loading data from: {input_path}")
        df = load_glucose_data(input_path)
        
        main_action.log(
            message_type="data_loaded",
            shape=df.shape,
            unique_sequences=df['unique_id'].n_unique(),
            date_range=f"{df['ds'].min()} to {df['ds'].max()}"
        )
        typer.echo(f"Data shape: {df.shape}")
        typer.echo(f"Unique sequences: {df['unique_id'].n_unique()}")
        typer.echo(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
        
        # Parse model names
        model_names_list = None
        if models_to_train:
            model_names_list = [name.strip() for name in models_to_train.split(',')]
        
        # Get list of model names to train
        available_models_dict = get_available_models(horizon, input_size, max_steps)
        
        if model_names_list is None:
            # Use default model selection
            model_names_list = get_default_models()
        
        main_action.log(
            message_type="training_config",
            total_models=len(model_names_list),
            models=model_names_list,
            horizon=horizon,
            input_size=input_size,
            max_steps=max_steps,
            n_windows=n_windows if test_size is None else None,
            test_size=test_size
        )
        
        typer.echo(f"\n{'='*70}")
        typer.echo(f"Training {len(model_names_list)} models iteratively")
        typer.echo(f"Models: {', '.join(model_names_list)}")
        typer.echo(f"{'='*70}\n")
        
        # Track results
        all_metrics: List[Dict[str, Any]] = []
        successful_models: List[str] = []
        failed_models: List[Dict[str, str]] = []
        
        # Iterate through each model
        for step, model_name in enumerate(model_names_list, 1):
            typer.echo(f"\n{'='*70}")
            typer.echo(f"Step {step}/{len(model_names_list)}: Training {model_name}")
            typer.echo(f"{'='*70}")
            
            if model_name not in available_models_dict:
                typer.echo(f"❌ Model {model_name} not available. Skipping...")
                failed_models.append({
                    'model': model_name,
                    'error': 'Model not available',
                    'step': step
                })
                continue
            
            try:
                with start_action(action_type=f"train_model", model=model_name, step=step) as action:
                    # Initialize single model
                    action.log(message_type="initializing_model")
                    model = available_models_dict[model_name]()
                    nf = NeuralForecast(models=[model], freq='5min')
                    
                    typer.echo(f"  🔄 Running cross-validation for {model_name}...")
                    action.log(message_type="starting_cross_validation")
                    
                    # Run cross-validation
                    if test_size is not None:
                        cv_df = nf.cross_validation(
                            df=df,
                            test_size=test_size,
                            step_size=horizon,
                            n_windows=None
                        )
                    else:
                        cv_df = nf.cross_validation(
                            df=df,
                            n_windows=n_windows,
                            step_size=horizon
                        )
                    
                    action.log(message_type="cross_validation_completed")
                    typer.echo(f"  ✅ Cross-validation completed for {model_name}")
                    
                    # Evaluate model
                    typer.echo(f"  📊 Evaluating {model_name}...")
                    action.log(message_type="evaluating_metrics")
                    metrics_list = [mae, mse, rmse, mape]
                    
                    # Convert to pandas for evaluation
                    if isinstance(cv_df, pl.DataFrame):
                        cv_df_pandas = cv_df.to_pandas()
                    else:
                        cv_df_pandas = cv_df
                    
                    cv_df_eval = cv_df_pandas.drop(columns=['cutoff'])
                    
                    # Evaluate
                    evaluation_df = evaluate(
                        cv_df_eval,
                        metrics=metrics_list,
                        models=[model_name]
                    )
                    
                    # Add step number to metrics
                    evaluation_df['step'] = step
                    all_metrics.append(evaluation_df)
                    
                    # Log metrics
                    metrics_dict = {}
                    for _, row in evaluation_df.iterrows():
                        metrics_dict[row['metric']] = float(row[model_name])
                    action.log(message_type="model_metrics", metrics=metrics_dict)
                    
                    # Save incremental metrics
                    metrics_path = output_path / 'metrics.csv'
                    if all_metrics:
                        combined_metrics = pd.concat(all_metrics, ignore_index=True)
                        pl.from_pandas(combined_metrics).write_csv(metrics_path)
                    
                    # Display metrics for this model
                    typer.echo(f"\n  Metrics for {model_name}:")
                    for _, row in evaluation_df.iterrows():
                        typer.echo(f"    {row['metric']}: {row[model_name]:.4f}")
                    
                    # Plot predictions
                    typer.echo(f"  📈 Creating prediction plots for {model_name}...")
                    action.log(message_type="creating_plots")
                    plot_predictions(df, cv_df_pandas, model_name, output_path)
                    
                    # Save model
                    typer.echo(f"  💾 Saving {model_name}...")
                    action.log(message_type="saving_model")
                    model_dir = models_path / model_name
                    model_dir.mkdir(parents=True, exist_ok=True)
                    nf.save(path=str(model_dir), overwrite=True)
                    
                    # Save model-specific CV results
                    cv_results_path = output_path / f'cv_results_{model_name}.csv'
                    if isinstance(cv_df, pl.DataFrame):
                        cv_df.write_csv(cv_results_path)
                    else:
                        pl.from_pandas(cv_df).write_csv(cv_results_path)
                    action.log(message_type="cv_results_saved", path=str(cv_results_path))
                    
                    successful_models.append(model_name)
                    action.log(message_type="model_completed", success=True)
                    typer.echo(f"  ✅ Step {step}/{len(model_names_list)}: {model_name} completed successfully!")
                    
            except Exception as e:
                error_msg = str(e)
                error_trace = traceback.format_exc()
                typer.echo(f"  ❌ Error training {model_name}: {error_msg}")
                typer.echo(f"  Traceback:\n{error_trace}")
                
                failed_models.append({
                    'model': model_name,
                    'error': error_msg,
                    'step': step
                })
                
                # Save error log
                error_log_path = output_path / f'error_{model_name}.txt'
                with open(error_log_path, 'w') as f:
                    f.write(f"Error training {model_name}:\n")
                    f.write(f"{error_trace}\n")
                
                typer.echo(f"  ⚠️  Continuing with next model...")
                continue
        
        # Final summary
        typer.echo(f"\n{'='*70}")
        typer.echo("TRAINING SUMMARY")
        typer.echo(f"{'='*70}")
        typer.echo(f"✅ Successful models ({len(successful_models)}): {', '.join(successful_models)}")
        
        if failed_models:
            typer.echo(f"\n❌ Failed models ({len(failed_models)}):")
            for fail in failed_models:
                typer.echo(f"  Step {fail['step']}: {fail['model']} - {fail['error']}")
        
        # Save final metrics summary
        if all_metrics:
            metrics_path = output_path / 'metrics.csv'
            combined_metrics = pd.concat(all_metrics, ignore_index=True)
            pl.from_pandas(combined_metrics).write_csv(metrics_path)
            typer.echo(f"\n📊 Final metrics saved to: {metrics_path}")
        
        # Save summary report
        summary_path = output_path / 'training_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total models attempted: {len(model_names_list)}\n")
            f.write(f"Successful: {len(successful_models)}\n")
            f.write(f"Failed: {len(failed_models)}\n\n")
            f.write(f"Successful models: {', '.join(successful_models)}\n\n")
            if failed_models:
                f.write("Failed models:\n")
                for fail in failed_models:
                    f.write(f"  Step {fail['step']}: {fail['model']} - {fail['error']}\n")
        
        typer.echo(f"📝 Summary saved to: {summary_path}")
        typer.echo("\n✨ Training complete!")


if __name__ == "__main__":
    app()