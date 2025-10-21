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
    get_models_by_category,
    get_models_supporting_exogenous
)
from glucose_neuralforecast.config import (
    load_config, 
    save_default_config, 
    save_config,
    TrainingConfig,
    generate_run_id,
    list_training_runs
)

app = typer.Typer()


@app.command()
def list_models() -> None:
    """
    List all available models that can be trained.
    """
    from glucose_neuralforecast.models import get_models_supporting_exogenous
    
    typer.echo("\n📊 Available NeuralForecast Models:\n")
    
    models_by_category = get_models_by_category()
    default_models = get_default_models()
    exog_models = get_models_supporting_exogenous()
    
    for category, models in models_by_category.items():
        typer.echo(f"\n{category}:")
        for model in models:
            exog_marker = " 🔗" if model in exog_models else ""
            typer.echo(f"  • {model}{exog_marker}")
    
    typer.echo(f"\n\n📝 Total: {sum(len(models) for models in models_by_category.values())} models available")
    typer.echo("🔗 = Supports exogenous variables")
    typer.echo("\nUsage: train --models \"NBEATS,NHITS,LSTM\"")
    typer.echo(f"\nDefault models (if --models not specified): {len(default_models)} models (ALL support exogenous variables 🔗)")
    typer.echo("  MLP-based: NHITS, NBEATSx, MLP, MLPMultivariate")
    typer.echo("  RNN-based: LSTM, GRU, RNN, DilatedRNN")
    typer.echo("  CNN-based: TCN, BiTCN")
    typer.echo("  Transformers: VanillaTransformer, Informer, Autoformer, FEDformer")
    typer.echo("  Specialized: TFT, DeepAR, DeepNPTS, TiDE, HINT")
    typer.echo("  Recent: TimesNet, TimeXer, TSMixerx")
    typer.echo("  KAN: KAN\n")


@app.command()
def generate_config(
    output_file: str = typer.Option(
        "train_config.yaml",
        "--output",
        "-o",
        help="Path for the generated config file"
    )
) -> None:
    """
    Generate a default training configuration YAML file.
    """
    output_path = Path(output_file)
    
    if output_path.exists():
        overwrite = typer.confirm(f"File {output_file} already exists. Overwrite?")
        if not overwrite:
            typer.echo("❌ Cancelled")
            return
    
    save_default_config(output_path)
    typer.echo(f"✅ Default configuration saved to: {output_file}")


@app.command()
def list_runs(
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Base output directory. If not provided, uses data/output"
    )
) -> None:
    """
    List all available training runs.
    """
    base = resolve_base_folder()
    data_dir = base / 'data'
    output_path = Path(output_dir) if output_dir else data_dir / 'output'
    
    runs = list_training_runs(output_path)
    
    if not runs:
        typer.echo("📭 No training runs found")
        typer.echo(f"   Runs are stored in: {output_path / 'runs'}")
        return
    
    typer.echo(f"\n📊 Available Training Runs ({len(runs)}):\n")
    typer.echo(f"{'Run ID':<25} {'Models':<8} {'Horizon':<8} {'Steps':<8} {'Status'}")
    typer.echo("=" * 80)
    
    for run in runs:
        run_id = run['run_id']
        models_count = len(run.get('models', [])) if 'models' in run else '?'
        horizon = run.get('horizon', '?')
        max_steps = run.get('max_steps', '?')
        
        status_parts = []
        if run['config_exists']:
            status_parts.append('📋 config')
        if run['models_exist']:
            status_parts.append('🤖 models')
        status = ' '.join(status_parts) if status_parts else '⚠️  incomplete'
        
        typer.echo(f"{run_id:<25} {str(models_count):<8} {str(horizon):<8} {str(max_steps):<8} {status}")
    
    typer.echo(f"\n💡 Use --run-id to specify a run for inference")
    typer.echo(f"   Example: predict --run-id {runs[0]['run_id']}\n")


@app.command()
def train_from_config(
    config_file: str = typer.Option(
        "train_config.yaml",
        "--config",
        "-c",
        help="Path to the YAML configuration file"
    )
) -> None:
    """
    Train models using configuration from a YAML file.
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        typer.echo(f"❌ Configuration file not found: {config_file}")
        typer.echo(f"Generate a default config with: generate-config")
        raise typer.Exit(1)
    
    typer.echo(f"📋 Loading configuration from: {config_file}")
    config = load_config(config_path)
    
    typer.echo(f"✅ Configuration loaded successfully")
    typer.echo(f"\nTraining parameters:")
    typer.echo(f"  Horizon: {config.horizon}")
    typer.echo(f"  Input size: {config.input_size}")
    typer.echo(f"  Max steps: {config.max_steps}")
    typer.echo(f"  Models: {', '.join(config.models)}")
    typer.echo(f"  N windows: {config.n_windows}")
    typer.echo(f"  Test size: {config.test_size}")
    
    # Call the train function with config parameters
    train(
        data_file=config.data_file,
        output_dir=config.output_dir,
        run_id=config.run_id,
        horizon=config.horizon,
        input_size=config.input_size,
        max_steps=config.max_steps,
        models_to_train=','.join(config.models),
        n_windows=config.n_windows,
        test_size=config.test_size,
        log_file=config.log_file
    )


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
        help="Base output directory. If not provided, uses data/output"
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Unique identifier for this training run. If not provided, a timestamp-based ID will be generated"
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
        help="Comma-separated list of models to train (e.g., 'NBEATS,NHITS,LSTM'). If not provided, trains all 24 models that support exogenous variables"
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
        
        # Generate run_id if not provided
        if run_id is None:
            run_id = generate_run_id()
        
        typer.echo(f"🎯 Training run ID: {run_id}")
        
        # Set up paths
        data_dir = base / 'data'
        input_path = Path(data_file) if data_file else data_dir / 'input' / 'livia_glucose.csv'
        base_output_path = Path(output_dir) if output_dir else data_dir / 'output'
        
        # Create run directory structure
        output_path = base_output_path / 'runs' / run_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        typer.echo(f"📁 Output directory: {output_path}")
        
        # Create models directory
        models_path = output_path / 'models'
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        logs_dir = base / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging with pycomfort
        to_nice_file(
            output_file=logs_dir / f'train_{max_steps}.json',
            rendered_file=logs_dir / f'train_{max_steps}.log'
        )
        
        main_action.log(message_type="logging_setup", logs_dir=str(logs_dir))
        typer.echo(f"Logs directory: {logs_dir}")
        
        # Parse model names
        model_names_list = None
        if models_to_train:
            model_names_list = [name.strip() for name in models_to_train.split(',')]
        
        # Get list of model names to train
        available_models_dict = get_available_models(horizon, input_size, max_steps)
        
        if model_names_list is None:
            # Use default model selection
            model_names_list = get_default_models()
        
        # Get models that support exogenous variables
        exogenous_capable_models = get_models_supporting_exogenous()
        
        # Create model configurations (some models will be trained twice: univariate and with exogenous)
        model_configs = []
        for model_name in model_names_list:
            # Always add univariate version
            model_configs.append({
                'model_name': model_name,
                'use_exogenous': False,
                'display_name': model_name
            })
            
            # Add exogenous version if model supports it
            if model_name in exogenous_capable_models:
                model_configs.append({
                    'model_name': model_name,
                    'use_exogenous': True,
                    'display_name': f"{model_name}_exog"
                })
        
        # Create and save training configuration
        training_config = TrainingConfig(
            run_id=run_id,
            data_file=str(input_path),
            output_dir=str(base_output_path),
            horizon=horizon,
            input_size=input_size,
            max_steps=max_steps,
            models=model_names_list,
            n_windows=n_windows,
            test_size=test_size,
            log_file=log_file
        )
        
        config_path = output_path / 'config.yaml'
        save_config(training_config, config_path)
        typer.echo(f"💾 Configuration saved to: {config_path}")
        
        typer.echo(f"\n{'='*70}")
        typer.echo(f"Model configurations to train: {len(model_configs)}")
        typer.echo(f"Base models: {len(model_names_list)}")
        typer.echo(f"Models with exogenous: {sum(1 for cfg in model_configs if cfg['use_exogenous'])}")
        typer.echo(f"{'='*70}\n")
        
        main_action.log(
            message_type="training_config",
            run_id=run_id,
            total_configs=len(model_configs),
            base_models=len(model_names_list),
            models=model_names_list,
            horizon=horizon,
            input_size=input_size,
            max_steps=max_steps,
            n_windows=n_windows if test_size is None else None,
            test_size=test_size
        )
        
        # Track results
        all_metrics: Dict[str, pl.DataFrame] = {}  # Store as dict with display name as key
        successful_models: List[str] = []
        failed_models: List[Dict[str, str]] = []
        
        # Iterate through each model configuration
        for step, config in enumerate(model_configs, 1):
            model_name = config['model_name']
            use_exogenous = config['use_exogenous']
            display_name = config['display_name']
            
            typer.echo(f"\n{'='*70}")
            typer.echo(f"Step {step}/{len(model_configs)}: Training {display_name}")
            if use_exogenous:
                typer.echo(f"  (with exogenous variables)")
            typer.echo(f"{'='*70}")
            
            if model_name not in available_models_dict:
                typer.echo(f"❌ Model {model_name} not available. Skipping...")
                failed_models.append({
                    'model': display_name,
                    'error': 'Model not available',
                    'step': step
                })
                continue
            
            try:
                with start_action(action_type=f"train_model", model=display_name, base_model=model_name, use_exogenous=use_exogenous, step=step) as action:
                    # Load data with or without exogenous variables
                    main_action.log(message_type="loading_data", path=str(input_path), use_exogenous=use_exogenous)
                    typer.echo(f"  📊 Loading data (exogenous: {use_exogenous})...")
                    df = load_glucose_data(input_path, include_exogenous=use_exogenous)
                    
                    action.log(
                        message_type="data_loaded",
                        shape=df.shape,
                        columns=df.columns,
                        unique_sequences=df['unique_id'].n_unique(),
                        date_range=f"{df['ds'].min()} to {df['ds'].max()}"
                    )
                    typer.echo(f"  Data shape: {df.shape}, columns: {df.columns}")
                    
                    # Initialize single model
                    action.log(message_type="initializing_model")
                    model = available_models_dict[model_name]()
                    nf = NeuralForecast(models=[model], freq='5min')
                    
                    typer.echo(f"  🔄 Running cross-validation for {display_name}...")
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
                    typer.echo(f"  ✅ Cross-validation completed for {display_name}")
                    
                    # Evaluate model
                    typer.echo(f"  📊 Evaluating {display_name}...")
                    action.log(message_type="evaluating_metrics")
                    metrics_list = [mae, mse, rmse, mape]
                    
                    # Convert to pandas for evaluation
                    if isinstance(cv_df, pl.DataFrame):
                        cv_df_pandas = cv_df.to_pandas()
                    else:
                        cv_df_pandas = cv_df
                    
                    cv_df_eval = cv_df_pandas.drop(columns=['cutoff'])
                    
                    # Evaluate (using base model_name as the column name in cv_df)
                    evaluation_df = evaluate(
                        cv_df_eval,
                        metrics=metrics_list,
                        models=[model_name]
                    )
                    
                    # Rename the model column to display_name for clarity
                    evaluation_df = evaluation_df.rename(columns={model_name: display_name})
                    
                    # Convert to polars and store
                    evaluation_pl = pl.from_pandas(evaluation_df)
                    all_metrics[display_name] = evaluation_pl
                    
                    # Log metrics
                    metrics_dict = {}
                    for _, row in evaluation_df.iterrows():
                        metrics_dict[row['metric']] = float(row[display_name])
                    action.log(message_type="model_metrics", metrics=metrics_dict)
                    
                    # Save incremental metrics by joining all models so far
                    metrics_path = output_path / 'metrics.csv'
                    if all_metrics:
                        # Start with the first model's metrics
                        combined_metrics = list(all_metrics.values())[0]
                        
                        # Join with each subsequent model
                        for i, model_df in enumerate(list(all_metrics.values())[1:], 1):
                            combined_metrics = combined_metrics.join(
                                model_df,
                                on=['unique_id', 'metric'],
                                how='outer',
                                coalesce=True
                            )
                        
                        combined_metrics.write_csv(metrics_path)
                    
                    # Display metrics for this model
                    typer.echo(f"\n  Metrics for {display_name}:")
                    for _, row in evaluation_df.iterrows():
                        typer.echo(f"    {row['metric']}: {row[display_name]:.4f}")
                    
                    # Plot predictions
                    typer.echo(f"  📈 Creating prediction plots for {display_name}...")
                    action.log(message_type="creating_plots")
                    plot_predictions(df, cv_df_pandas, display_name, output_path)
                    
                    # Save model
                    typer.echo(f"  💾 Saving {display_name}...")
                    action.log(message_type="saving_model")
                    model_dir = models_path / display_name
                    model_dir.mkdir(parents=True, exist_ok=True)
                    nf.save(path=str(model_dir), overwrite=True)
                    
                    # Save model-specific CV results
                    cv_results_path = output_path / f'cv_results_{display_name}.csv'
                    if isinstance(cv_df, pl.DataFrame):
                        cv_df.write_csv(cv_results_path)
                    else:
                        pl.from_pandas(cv_df).write_csv(cv_results_path)
                    action.log(message_type="cv_results_saved", path=str(cv_results_path))
                    
                    successful_models.append(display_name)
                    action.log(message_type="model_completed", success=True)
                    typer.echo(f"  ✅ Step {step}/{len(model_configs)}: {display_name} completed successfully!")
                    
            except Exception as e:
                error_msg = str(e)
                error_trace = traceback.format_exc()
                typer.echo(f"  ❌ Error training {display_name}: {error_msg}")
                typer.echo(f"  Traceback:\n{error_trace}")
                
                failed_models.append({
                    'model': display_name,
                    'error': error_msg,
                    'step': step
                })
                
                # Save error log
                error_log_path = output_path / f'error_{display_name}.txt'
                with open(error_log_path, 'w') as f:
                    f.write(f"Error training {display_name}:\n")
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
            
            # Join all model metrics by unique_id and metric
            combined_metrics = list(all_metrics.values())[0]
            for model_df in list(all_metrics.values())[1:]:
                combined_metrics = combined_metrics.join(
                    model_df,
                    on=['unique_id', 'metric'],
                    how='outer',
                    coalesce=True
                )
            
            combined_metrics.write_csv(metrics_path)
            typer.echo(f"\n📊 Final metrics saved to: {metrics_path}")
            
            # Also save a summary with aggregated metrics per model
            metrics_summary_path = output_path / 'metrics_summary.csv'
            
            # Get list of model columns (exclude unique_id and metric)
            model_cols = [col for col in combined_metrics.columns if col not in ['unique_id', 'metric']]
            
            # Create summary: for each metric, aggregate across all unique_ids
            summary_rows = []
            for metric_name in combined_metrics['metric'].unique():
                metric_data = combined_metrics.filter(pl.col('metric') == metric_name)
                row = {'metric': metric_name}
                for model_col in model_cols:
                    if model_col in metric_data.columns:
                        row[f'{model_col}_mean'] = metric_data[model_col].mean()
                        row[f'{model_col}_std'] = metric_data[model_col].std()
                summary_rows.append(row)
            
            summary_df = pl.DataFrame(summary_rows)
            summary_df.write_csv(metrics_summary_path)
            typer.echo(f"📊 Metrics summary saved to: {metrics_summary_path}")
        
        # Save summary report
        summary_path = output_path / 'training_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total configurations attempted: {len(model_configs)}\n")
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