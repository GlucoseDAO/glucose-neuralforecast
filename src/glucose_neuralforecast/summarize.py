from pathlib import Path
from typing import Optional

import polars as pl
import typer
from eliot import start_action

from glucose_neuralforecast.utils import resolve_base_folder

app = typer.Typer()


@app.command()
def combine_metrics(
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory containing model CV results. If not provided, uses data/output"
    )
) -> None:
    """
    Combine metrics from individual model cv_results_*.csv files into a single joined metrics.csv.
    This joins results by unique_id and metric columns, creating one row per unique_id/metric combination
    with columns for each model.
    """
    with start_action(action_type="combine_metrics") as action:
        # Resolve base folder
        base = resolve_base_folder()
        output_path = Path(output_dir) if output_dir else base / 'data' / 'output'
        
        action.log(message_type="output_directory", path=str(output_path))
        typer.echo(f"Output directory: {output_path}")
        
        # Find all cv_results files
        cv_files = sorted(output_path.glob('cv_results_*.csv'))
        
        if not cv_files:
            typer.echo("❌ No cv_results_*.csv files found in output directory")
            return
        
        action.log(message_type="found_files", count=len(cv_files), files=[f.name for f in cv_files])
        typer.echo(f"Found {len(cv_files)} cv_results files")
        
        # Extract model names from filenames
        model_names = [f.stem.replace('cv_results_', '') for f in cv_files]
        typer.echo(f"Models: {', '.join(model_names)}")
        
        # Read and process each file
        all_metrics = {}
        
        for cv_file, model_name in zip(cv_files, model_names):
            typer.echo(f"  Processing {model_name}...")
            action.log(message_type="processing_model", model=model_name, file=cv_file.name)
            
            # Read CV results
            cv_df = pl.read_csv(cv_file)
            
            # Check what columns are available
            required_cols = {'unique_id', 'ds', 'y', model_name}
            if not required_cols.issubset(set(cv_df.columns)):
                typer.echo(f"  ⚠️  Skipping {model_name}: missing required columns")
                continue
            
            # Calculate metrics for each unique_id
            # Group by unique_id and calculate MAE, MSE, RMSE, MAPE
            metrics_df = cv_df.group_by('unique_id').agg([
                ((pl.col('y') - pl.col(model_name)).abs().mean()).alias('mae'),
                ((pl.col('y') - pl.col(model_name)).pow(2).mean()).alias('mse'),
                ((pl.col('y') - pl.col(model_name)).pow(2).mean().sqrt()).alias('rmse'),
                ((pl.col('y') - pl.col(model_name)).abs() / pl.col('y').abs() * 100).mean().alias('mape')
            ])
            
            # Transform to long format with metric column
            metrics_long = metrics_df.unpivot(
                index='unique_id',
                on=['mae', 'mse', 'rmse', 'mape'],
                variable_name='metric',
                value_name=model_name
            )
            
            all_metrics[model_name] = metrics_long
            action.log(message_type="processed_model", model=model_name, rows=len(metrics_long))
        
        if not all_metrics:
            typer.echo("❌ No valid metrics data found")
            return
        
        typer.echo("\n📊 Combining metrics from all models...")
        action.log(message_type="combining_metrics", models=list(all_metrics.keys()))
        
        # Join all metrics by unique_id and metric
        combined_metrics = list(all_metrics.values())[0]
        for model_name, model_df in list(all_metrics.items())[1:]:
            typer.echo(f"  Joining {model_name}...")
            combined_metrics = combined_metrics.join(
                model_df,
                on=['unique_id', 'metric'],
                how='outer',
                coalesce=True
            )
        
        # Save combined metrics
        metrics_path = output_path / 'metrics.csv'
        combined_metrics.write_csv(metrics_path)
        typer.echo(f"\n✅ Combined metrics saved to: {metrics_path}")
        action.log(message_type="metrics_saved", path=str(metrics_path), rows=len(combined_metrics))
        
        # Create summary with mean and std for each metric and model
        metrics_summary_path = output_path / 'metrics_summary.csv'
        
        # Get list of model columns (exclude unique_id and metric)
        model_cols = [col for col in combined_metrics.columns if col not in ['unique_id', 'metric']]
        
        typer.echo("\n📈 Creating metrics summary...")
        
        # Create summary: for each model, aggregate metrics across all unique_ids
        summary_rows = []
        for model_col in model_cols:
            row = {'model': model_col}
            for metric_name in sorted(combined_metrics['metric'].unique()):
                metric_data = combined_metrics.filter(
                    (pl.col('metric') == metric_name) & (pl.col(model_col).is_not_null())
                )
                if len(metric_data) > 0:
                    # Get mean value for this metric across all unique_ids
                    mean_val = metric_data[model_col].mean()
                    row[metric_name] = float(mean_val) if mean_val is not None else None
            summary_rows.append(row)
        
        summary_df = pl.DataFrame(summary_rows)
        
        # Ensure metric columns are float type and sort by MAE
        metric_cols = [col for col in summary_df.columns if col not in ['model']]
        for col in metric_cols:
            summary_df = summary_df.with_columns(pl.col(col).cast(pl.Float64))
        
        # Sort by MAE in ascending order
        if 'mae' in summary_df.columns:
            summary_df = summary_df.sort('mae')
        
        summary_df.write_csv(metrics_summary_path)
        typer.echo(f"✅ Metrics summary saved to: {metrics_summary_path}")
        action.log(message_type="summary_saved", path=str(metrics_summary_path))
        
        # Display summary
        typer.echo("\n" + "="*70)
        typer.echo("METRICS SUMMARY")
        typer.echo("="*70)
        typer.echo(summary_df)
        typer.echo("\n✨ Done!")


if __name__ == "__main__":
    app()
