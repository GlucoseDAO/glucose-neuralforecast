"""Data loading and preparation functions."""

from pathlib import Path

import polars as pl
from eliot import start_action


def load_glucose_data(input_path: Path) -> pl.DataFrame:
    """
    Load and prepare glucose data for NeuralForecast.
    
    NeuralForecast expects columns: unique_id, ds, y
    
    Args:
        input_path: Path to the CSV file containing glucose data
        
    Returns:
        pl.DataFrame: DataFrame with columns unique_id, ds, y ready for NeuralForecast
    """
    with start_action(action_type="load_glucose_data", input_path=str(input_path)) as action:
        # Read CSV with polars
        df = pl.read_csv(input_path)
        
        action.log(message_type="raw_data_loaded", shape=df.shape)
        
        # Filter only EGV (Estimated Glucose Value) events
        df = df.filter(pl.col("Event Type") == "EGV")
        
        # Prepare data in NeuralForecast format
        df_forecast = df.select([
            pl.col("sequence_id").alias("unique_id"),
            pl.col("Timestamp (YYYY-MM-DDThh:mm:ss)").str.to_datetime().alias("ds"),
            pl.col("Glucose Value (mg/dL)").alias("y")
        ])
        
        # Drop any rows with null values in y
        df_forecast = df_forecast.drop_nulls(subset=["y"])
        
        action.log(
            message_type="data_prepared",
            shape=df_forecast.shape,
            unique_sequences=df_forecast["unique_id"].n_unique()
        )
        
        return df_forecast

