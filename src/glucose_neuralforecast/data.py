"""Data loading and preparation functions."""

from pathlib import Path

import polars as pl
from eliot import start_action


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
        base_columns = [
            pl.col("sequence_id").alias("unique_id"),
            pl.col("Timestamp (YYYY-MM-DDThh:mm:ss)").str.to_datetime().alias("ds"),
            pl.col("Glucose Value (mg/dL)").alias("y")
        ]
        
        # Add exogenous variables if requested
        if include_exogenous:
            exog_columns = [
                pl.col("Fast-Acting Insulin Value (u)").fill_null(0.0).alias("fast_insulin"),
                pl.col("Long-Acting Insulin Value (u)").fill_null(0.0).alias("long_insulin"),
                pl.col("Carb Value (grams)").fill_null(0.0).alias("carbs"),
                pl.col("flow_amount").fill_null(0.0).alias("flow_amount")
            ]
            df_forecast = df.select(base_columns + exog_columns)
        else:
            df_forecast = df.select(base_columns)
        
        # Drop any rows with null values in y
        df_forecast = df_forecast.drop_nulls(subset=["y"])
        
        action.log(
            message_type="data_prepared",
            shape=df_forecast.shape,
            unique_sequences=df_forecast["unique_id"].n_unique(),
            columns=df_forecast.columns,
            include_exogenous=include_exogenous
        )
        
        return df_forecast

