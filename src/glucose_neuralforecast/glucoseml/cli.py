"""Command-line interface for GlucoseML dataset integration."""

from pathlib import Path
from typing import Optional, List

import typer
from eliot import start_action

from glucose_neuralforecast.utils import resolve_base_folder
from glucose_neuralforecast.glucoseml.registry import load_registry, list_datasets
from glucose_neuralforecast.glucoseml.download import download_all_datasets
from glucose_neuralforecast.glucoseml.preprocess import preprocess_all_datasets
from glucose_neuralforecast.glucoseml.ingest import ingest_for_neuralforecast, get_neuralforecast_summary

app = typer.Typer(help="GlucoseML dataset integration commands")


def parse_dataset_list(datasets_str: str) -> List[str]:
    """Parse comma-separated dataset list or 'all'."""
    if datasets_str.lower() == "all":
        return list_datasets()
    return [d.strip() for d in datasets_str.split(',') if d.strip()]


@app.command()
def list_available() -> None:
    """
    List all available datasets in the registry.
    """
    typer.echo("\n📊 Available GlucoseML Datasets:\n")
    
    registry = load_registry()
    
    for dataset_name, dataset_config in registry.datasets.items():
        typer.echo(f"  • {dataset_name}")
        typer.echo(f"    Name: {dataset_config.name}")
        typer.echo(f"    Source: {dataset_config.source}")
        typer.echo(f"    Format: {dataset_config.format}")
        typer.echo(f"    Sensor: {dataset_config.sensor_family}")
        typer.echo(f"    Sampling: {dataset_config.sampling_minutes} minutes")
        typer.echo(f"    License: {dataset_config.license_note}")
        typer.echo()
    
    typer.echo(f"Total: {len(registry.datasets)} datasets\n")


@app.command()
def download(
    datasets: str = typer.Option(
        "all",
        "--datasets",
        "-d",
        help="Comma-separated dataset names or 'all'"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for raw data. Default: data/input/glucoseml/raw"
    )
) -> None:
    """
    Download raw datasets from their sources.
    """
    with start_action(action_type="cli_download", datasets_arg=datasets):
        base = resolve_base_folder()
        out_path = Path(output_dir) if output_dir else base / 'data' / 'input' / 'glucoseml' / 'raw'
        dataset_list = parse_dataset_list(datasets)
        
        download_all_datasets(dataset_list, out_path)
        typer.echo(f"✅ Downloaded {len(dataset_list)} datasets to {out_path}")


@app.command()
def preprocess(
    datasets: str = typer.Option(
        "all",
        "--datasets",
        "-d",
        help="Comma-separated dataset names or 'all'"
    ),
    input_dir: Optional[str] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input directory for raw data. Default: data/input/glucoseml/raw"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for parquet files. Default: data/input/glucoseml/processed"
    ),
    no_validation: bool = typer.Option(
        False,
        "--no-validation",
        help="Skip validation (range, rate-of-change, coverage)"
    )
) -> None:
    """
    Preprocess raw datasets to parquet (one file per dataset) with inline validation and covariates.
    Outputs: {DATASET}.parquet + stats/per_dataset_counts.csv + stats/stats.yaml
    """
    with start_action(action_type="cli_preprocess", datasets_arg=datasets) as action:
        base = resolve_base_folder()
        in_path = Path(input_dir) if input_dir else base / 'data' / 'input' / 'glucoseml' / 'raw'
        out_path = Path(output_dir) if output_dir else base / 'data' / 'input' / 'glucoseml' / 'processed'
        dataset_list = parse_dataset_list(datasets)
        
        from glucose_neuralforecast.glucoseml.registry import load_registry
        registry = load_registry()
        
        all_stats = preprocess_all_datasets(
            dataset_list,
            in_path,
            out_path,
            registry_config=registry,
            apply_validation=not no_validation
        )
        
        typer.echo(f"✅ Preprocessed {len(all_stats)} datasets to {out_path}")


@app.command()
def ingest(
    source: str = typer.Option(
        "combined",
        "--from",
        "-f",
        help="Data source: 'combined', 'validated', or 'preprocessed'"
    ),
    datasets: Optional[str] = typer.Option(
        None,
        "--datasets",
        "-d",
        help="Comma-separated dataset names to include (optional filter)"
    ),
    output_format: str = typer.Option(
        "parquet",
        "--format",
        help="Output format: 'parquet' or 'csv'"
    ),
    output_path: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path. Default: data/input/glucoseml.parquet"
    ),
    include_metadata: bool = typer.Option(
        False,
        "--metadata/--no-metadata",
        help="Include metadata columns (dataset, sensor_family, etc.)"
    )
) -> None:
    """
    Ingest data for NeuralForecast training (exports unique_id, ds, y format).
    """
    with start_action(action_type="cli_ingest", source=source):
        base = resolve_base_folder()
        ext = "parquet" if output_format == "parquet" else "csv"
        out_file = Path(output_path) if output_path else base / 'data' / 'input' / f'glucoseml.{ext}'
        dataset_filter = parse_dataset_list(datasets) if datasets else None
        
        df = ingest_for_neuralforecast(
            source=source,
            output_path=out_file,
            output_format=output_format,
            datasets=dataset_filter,
            include_metadata=include_metadata
        )
        
        summary = get_neuralforecast_summary(df)
        typer.echo(f"✅ Ingested {summary['total_rows']} rows, {summary['unique_subjects']} subjects to {out_file}")


@app.command()
def pipeline(
    datasets: str = typer.Option(
        "all",
        "--datasets",
        "-d",
        help="Comma-separated dataset names or 'all'"
    ),
    skip_download: bool = typer.Option(
        False,
        "--skip-download",
        help="Skip download step (data already downloaded)"
    ),
    no_validation: bool = typer.Option(
        False,
        "--no-validation",
        help="Skip validation (range, rate-of-change, coverage)"
    )
) -> None:
    """
    Run the full pipeline: download → preprocess (with inline validation) → parquet output.
    """
    with start_action(action_type="cli_pipeline", datasets_arg=datasets):
        dataset_list = parse_dataset_list(datasets)
        
        if not skip_download:
            download(datasets=datasets, output_dir=None)
        
        preprocess(
            datasets=datasets,
            input_dir=None,
            output_dir=None,
            no_validation=no_validation
        )
        
        typer.echo(f"✅ Pipeline complete for {len(dataset_list)} datasets")


if __name__ == "__main__":
    app()

