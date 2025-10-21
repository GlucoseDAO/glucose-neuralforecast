"""Data loading and preparation functions."""

from pathlib import Path
from typing import List

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

