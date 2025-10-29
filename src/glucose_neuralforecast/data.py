"""Data loading and preparation functions."""

from pathlib import Path
from typing import List, Optional

import polars as pl
from eliot import start_action


def get_exogenous_columns() -> List[str]:
    """
    Get the list of exogenous variable column names.
    
    These are the columns used as historical exogenous variables (hist_exog_list)
    for models that support them.
    
    Returns:
        List[str]: List of exogenous column names
    """
    return ['fast_insulin', 'long_insulin', 'carbs', 'flow_amount']


def load_glucose_data(input_path: Path, include_exogenous: bool = False) -> pl.DataFrame:
    """
    Load and prepare glucose data for NeuralForecast.
    
    NeuralForecast expects columns: unique_id, ds, y
    For exogenous variables, additional columns are included.
    
    Args:
        input_path: Path to the CSV file containing glucose data
        include_exogenous: If True, includes exogenous variables (insulin, carbs, flow_amount)
        
    Returns:
        pl.DataFrame: DataFrame with columns unique_id, ds, y (and exogenous if requested) ready for NeuralForecast
    """
    with start_action(action_type="load_glucose_data", input_path=str(input_path), include_exogenous=include_exogenous) as action:
        # Read CSV with polars
        df = pl.read_csv(input_path)
        
        action.log(message_type="raw_data_loaded", shape=df.shape)
        
        # Filter only EGV (Estimated Glucose Value) events
        df = df.filter(pl.col("Event Type") == "EGV")
        
        # Base columns for NeuralForecast
        # Ensure proper types for all columns
        base_columns = [
            pl.col("sequence_id").cast(pl.Int64).alias("unique_id"),
            pl.col("Timestamp (YYYY-MM-DDThh:mm:ss)").str.to_datetime().alias("ds"),
            pl.col("Glucose Value (mg/dL)").cast(pl.Float64).alias("y")
        ]
        
        # Add exogenous variables if requested
        if include_exogenous:
            exog_columns = []
            for col_name, alias in [
                ("Fast-Acting Insulin Value (u)", "fast_insulin"),
                ("Long-Acting Insulin Value (u)", "long_insulin"),
                ("Carb Value (grams)", "carbs"),
                ("flow_amount", "flow_amount")
            ]:
                if col_name in df.columns:
                    # Explicitly handle type conversion: cast to string first, then to float
                    # This handles cases where the column might have mixed types
                    exog_columns.append(
                        pl.col(col_name)
                        .cast(pl.String)
                        .str.replace_all(",", "")  # Remove commas if present
                        .cast(pl.Float64, strict=False)  # Non-strict to handle any conversion issues
                        .fill_null(0.0)
                        .alias(alias)
                    )
                else:
                    action.log(message_type="warning", missing_column=col_name)
            df_forecast = df.select(base_columns + exog_columns)
        else:
            df_forecast = df.select(base_columns)
        
        # Drop any rows with null values in y
        df_forecast = df_forecast.drop_nulls(subset=["y"])
        
        # Ensure all numeric columns are Float64 (not object/string)
        for col in df_forecast.columns:
            if col not in ['unique_id', 'ds']:
                if df_forecast[col].dtype != pl.Float64:
                    df_forecast = df_forecast.with_columns(
                        pl.col(col).cast(pl.Float64)
                    )
        
        action.log(
            message_type="data_prepared",
            shape=df_forecast.shape,
            unique_sequences=df_forecast["unique_id"].n_unique(),
            columns=df_forecast.columns,
            include_exogenous=include_exogenous
        )
        
        return df_forecast


def load_glucoseml_data(
    input_path: Path,
    datasets: Optional[List[str]] = None,
    max_gap_minutes: int = 60,
    min_episode_length: int = 48,
    include_exogenous: bool = False
) -> pl.DataFrame:
    """
    Load and prepare GlucoseML parquet data for NeuralForecast.
    
    This function handles multiple subjects with potentially discontinuous time series
    by splitting each subject's data into episodes based on time gaps. Each episode
    becomes a separate time series with a unique_id like "dataset__subject_id__ep0".
    
    Args:
        input_path: Path to the GlucoseML processed directory or specific parquet file
        datasets: Optional list of dataset names to load (e.g., ['BIG_IDEAS', 'CGMacros']).
                 If None, loads all available parquet files.
        max_gap_minutes: Maximum time gap in minutes. Gaps larger than this split the series
                        into separate episodes (default: 60 minutes)
        min_episode_length: Minimum number of data points required for an episode to be kept
                           (default: 48 points, which is 4 hours at 5-minute intervals)
        include_exogenous: If True, includes available exogenous variables (carbs, insulin, etc.)
        
    Returns:
        pl.DataFrame: DataFrame with columns unique_id, ds, y (and exogenous if requested)
                     ready for NeuralForecast. Each unique_id represents one continuous episode.
    
    Example:
        >>> from pathlib import Path
        >>> # Load all datasets
        >>> df = load_glucoseml_data(Path("data/input/glucoseml/processed"))
        >>> 
        >>> # Load specific datasets only
        >>> df = load_glucoseml_data(
        ...     Path("data/input/glucoseml/processed"),
        ...     datasets=['BIG_IDEAS', 'ShanghaiT1DM']
        ... )
        >>> 
        >>> # With exogenous variables
        >>> df = load_glucoseml_data(
        ...     Path("data/input/glucoseml/processed"),
        ...     include_exogenous=True
        ... )
    """
    with start_action(
        action_type="load_glucoseml_data",
        input_path=str(input_path),
        datasets=datasets,
        max_gap_minutes=max_gap_minutes,
        min_episode_length=min_episode_length,
        include_exogenous=include_exogenous
    ) as action:
        
        # Determine files to load
        if input_path.is_file():
            # Single parquet file
            parquet_files = [input_path]
            action.log(message_type="loading_single_file", file=str(input_path))
        else:
            # Directory with multiple parquet files
            if datasets is None:
                # Load all parquet files (exclude stats subdirectory)
                parquet_files = [f for f in input_path.glob("*.parquet") if f.is_file()]
                action.log(message_type="loading_all_files", count=len(parquet_files))
            else:
                # Load specific datasets
                parquet_files = [input_path / f"{ds}.parquet" for ds in datasets]
                parquet_files = [f for f in parquet_files if f.exists()]
                action.log(message_type="loading_selected_files", files=[f.name for f in parquet_files])
        
        if not parquet_files:
            raise ValueError(f"No parquet files found at {input_path}")
        
        # Load and combine all datasets
        all_dfs = []
        for parquet_file in parquet_files:
            df = pl.read_parquet(parquet_file)
            all_dfs.append(df)
            action.log(
                message_type="loaded_parquet",
                file=parquet_file.name,
                rows=len(df),
                subjects=df["unique_id"].n_unique() if "unique_id" in df.columns else 0
            )
        
        # Align schemas across datasets (different datasets have different columns)
        # Collect all unique columns
        all_cols = set()
        for df in all_dfs:
            all_cols.update(df.columns)
        
        # Core columns that all datasets must have
        core_cols = {"dataset", "subject_id", "unique_id", "ds", "y", "sensor_family", "sampling_minutes", "timezone"}
        covariate_cols = sorted(all_cols - core_cols)
        
        action.log(message_type="schema_alignment", core_cols=list(core_cols), covariate_cols=covariate_cols)
        
        # Align all dataframes to have the same columns
        aligned_dfs = []
        for df in all_dfs:
            # Add missing covariate columns as nulls
            for cov_col in covariate_cols:
                if cov_col not in df.columns:
                    df = df.with_columns([
                        pl.lit(None).cast(pl.Float64).alias(cov_col)
                    ])
            
            # Reorder columns for consistency (core first, then covariates alphabetically)
            ordered_cols = sorted(core_cols) + covariate_cols
            df = df.select(ordered_cols)
            aligned_dfs.append(df)
        
        # Concatenate all datasets
        df_combined = pl.concat(aligned_dfs, how="vertical")
        action.log(
            message_type="combined_datasets",
            total_rows=len(df_combined),
            total_subjects=df_combined["unique_id"].n_unique()
        )
        
        # The GlucoseML parquet files already have unique_id format: "dataset__subject_id"
        # We need to split these into episodes based on time gaps
        
        # Sort by original unique_id and timestamp
        df_combined = df_combined.sort(["unique_id", "ds"])
        
        # Calculate time gaps within each subject
        df_combined = df_combined.with_columns([
            pl.col("ds").diff().over("unique_id").alias("time_gap")
        ])
        
        # Mark episode breaks (gaps > threshold or first row of each subject)
        df_combined = df_combined.with_columns([
            (
                pl.col("time_gap").is_null() |  # First row of each subject
                (pl.col("time_gap") > pl.duration(minutes=max_gap_minutes))
            ).cast(pl.Int32).cum_sum().over("unique_id").alias("episode")
        ])
        
        # Create new unique_id with episode number: "dataset__subject_id__ep0"
        df_combined = df_combined.with_columns([
            (pl.col("unique_id") + "__ep" + pl.col("episode").cast(pl.Utf8)).alias("unique_id_with_episode")
        ])
        
        # Count points per episode
        episode_counts = df_combined.group_by("unique_id_with_episode").agg([
            pl.count().alias("episode_length")
        ])
        
        # Filter episodes that are too short
        valid_episodes = episode_counts.filter(
            pl.col("episode_length") >= min_episode_length
        )["unique_id_with_episode"]
        
        df_filtered = df_combined.filter(
            pl.col("unique_id_with_episode").is_in(valid_episodes)
        )
        
        episodes_before = df_combined["unique_id_with_episode"].n_unique()
        episodes_after = df_filtered["unique_id_with_episode"].n_unique()
        
        action.log(
            message_type="episode_filtering",
            episodes_before=episodes_before,
            episodes_after=episodes_after,
            episodes_removed=episodes_before - episodes_after,
            rows_before=len(df_combined),
            rows_after=len(df_filtered)
        )
        
        # Replace unique_id with the episode-aware version
        df_filtered = df_filtered.with_columns([
            pl.col("unique_id_with_episode").alias("unique_id")
        ]).drop(["unique_id_with_episode", "time_gap", "episode"])
        
        # Prepare columns for NeuralForecast
        # Base required columns
        select_cols = ["unique_id", "ds", "y"]
        
        # Add exogenous variables if requested
        if include_exogenous:
            exog_mapping = {
                "Carb Value (grams)": "carbs",
                "Insulin Value (u)": "insulin",
                "Glucose Rate of Change (mg/dL/min)": "glucose_rate"
            }
            
            for orig_col, new_col in exog_mapping.items():
                if orig_col in df_filtered.columns:
                    df_filtered = df_filtered.with_columns([
                        pl.col(orig_col).fill_null(0.0).alias(new_col)
                    ])
                    select_cols.append(new_col)
                    action.log(message_type="added_exogenous", column=new_col)
        
        # Select final columns
        df_forecast = df_filtered.select(select_cols)
        
        # Drop any rows with null values in y
        df_forecast = df_forecast.drop_nulls(subset=["y"])
        
        # Ensure all numeric columns are Float64
        for col in df_forecast.columns:
            if col not in ['unique_id', 'ds']:
                if df_forecast[col].dtype != pl.Float64:
                    df_forecast = df_forecast.with_columns([
                        pl.col(col).cast(pl.Float64)
                    ])
        
        # Ensure unique_id is string type for NeuralForecast
        df_forecast = df_forecast.with_columns([
            pl.col("unique_id").cast(pl.Utf8)
        ])
        
        action.log(
            message_type="data_prepared",
            shape=df_forecast.shape,
            unique_episodes=df_forecast["unique_id"].n_unique(),
            columns=df_forecast.columns,
            include_exogenous=include_exogenous
        )
        
        return df_forecast

