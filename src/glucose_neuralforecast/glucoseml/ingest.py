"""Ingest data for NeuralForecast training."""

from pathlib import Path
from typing import Optional, List, Literal

import polars as pl
from eliot import start_action


def prepare_neuralforecast_data(
    df: pl.DataFrame,
    include_metadata: bool = False
) -> pl.DataFrame:
    """
    Prepare data for NeuralForecast by selecting required columns.
    
    Args:
        df: DataFrame with unified schema
        include_metadata: If True, keep dataset, subject_id, sensor_family, etc.
    
    Returns:
        DataFrame with NeuralForecast-ready format (unique_id, ds, y + optional metadata)
    """
    if include_metadata:
        return df
    else:
        return df.select(["unique_id", "ds", "y"])


def ingest_for_neuralforecast(
    source: str,
    output_path: Path,
    output_format: Literal["parquet", "csv"] = "parquet",
    datasets: Optional[List[str]] = None,
    include_metadata: bool = False
) -> pl.DataFrame:
    """
    Load preprocessed parquet files and export for NeuralForecast.
    
    Args:
        source: Not used (kept for compatibility)
        output_path: Path to save output file
        output_format: Output format ('parquet' or 'csv')
        datasets: Optional list of dataset names to include
        include_metadata: If True, keep all metadata columns
    
    Returns:
        DataFrame ready for NeuralForecast
    """
    with start_action(
        action_type="ingest_for_neuralforecast",
        output=str(output_path),
        datasets=datasets
    ) as action:
        from glucose_neuralforecast.utils import resolve_base_folder
        base = resolve_base_folder()
        data_dir = base / 'data' / 'input' / 'glucoseml' / 'processed'
        
        # Load parquet files
        all_dfs = []
        
        if datasets is None:
            # Load all parquet files
            parquet_files = list(data_dir.glob("*.parquet"))
            action.log(message_type="loading_all_parquets", count=len(parquet_files))
        else:
            # Load specific datasets
            parquet_files = [data_dir / f"{ds}.parquet" for ds in datasets]
            action.log(message_type="loading_selected_parquets", datasets=datasets)
        
        for parquet_file in parquet_files:
            if parquet_file.exists():
                df = pl.read_parquet(parquet_file)
                all_dfs.append(df)
                action.log(message_type="loaded_parquet", file=str(parquet_file), rows=len(df))
            else:
                action.log(message_type="parquet_not_found", file=str(parquet_file))
        
        if not all_dfs:
            raise ValueError("No parquet files found to load")
        
        # Align schemas across datasets (different datasets have different covariates)
        # Collect all unique columns
        all_cols = set()
        for df in all_dfs:
            all_cols.update(df.columns)
        
        # Core columns that all datasets must have
        core_cols = {"dataset", "subject_id", "unique_id", "ds", "y", "sensor_family", "sampling_minutes", "timezone"}
        covariate_cols = sorted(all_cols - core_cols)
        
        # Align all dataframes to have the same columns
        aligned_dfs = []
        for df in all_dfs:
            # Add missing covariate columns as nulls
            for cov_col in covariate_cols:
                if cov_col not in df.columns:
                    df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(cov_col))
            
            # Reorder columns for consistency
            ordered_cols = list(core_cols) + covariate_cols
            df = df.select(ordered_cols)
            aligned_dfs.append(df)
        
        # Concatenate all datasets
        df_combined = pl.concat(aligned_dfs)
        action.log(message_type="combined_data", total_rows=len(df_combined))
        
        # Prepare for NeuralForecast
        df_nf = prepare_neuralforecast_data(df_combined, include_metadata)
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "parquet":
            df_nf.write_parquet(output_path)
        else:
            df_nf.write_csv(output_path)
        
        action.log(
            message_type="saved_output",
            path=str(output_path),
            format=output_format,
            rows=len(df_nf)
        )
        
        return df_nf


def get_neuralforecast_summary(df: pl.DataFrame) -> dict:
    """
    Generate summary statistics for NeuralForecast data.
    
    Args:
        df: NeuralForecast DataFrame with unique_id, ds, y columns
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_rows": len(df),
        "unique_subjects": df["unique_id"].n_unique(),
        "date_range": f"{df['ds'].min()} to {df['ds'].max()}",
        "glucose_stats": {
            "min": df["y"].min(),
            "max": df["y"].max(),
            "mean": df["y"].mean(),
            "std": df["y"].std()
        }
    }
    
    # Add per-dataset stats if dataset column exists
    if "dataset" in df.columns:
        by_dataset = {}
        for dataset_name in df["dataset"].unique():
            df_ds = df.filter(pl.col("dataset") == dataset_name)
            by_dataset[dataset_name] = {
                "rows": len(df_ds),
                "subjects": df_ds["unique_id"].n_unique()
            }
        summary["by_dataset"] = by_dataset
    
    return summary
