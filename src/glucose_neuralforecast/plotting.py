"""Plotting functions for visualizing model predictions."""

from pathlib import Path

import polars as pl
import pandas as pd
from eliot import start_action
from utilsforecast.plotting import plot_series


def plot_predictions(
    df: pl.DataFrame,
    cv_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
    max_sequences: int = 3
) -> None:
    """
    Plot prediction examples for a model and save to file using utilsforecast.
    
    Args:
        df: Original data (polars)
        cv_df: Cross-validation results with predictions (pandas)
        model_name: Name of the model
        output_path: Directory to save plots
        max_sequences: Maximum number of sequences to plot
    """
    plots_dir = output_path / 'plots' / model_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    with start_action(action_type="plot_predictions", model=model_name) as action:
        try:
            # Convert to pandas for plot_series
            df_pandas = df.to_pandas() if isinstance(df, pl.DataFrame) else df
            
            # Get unique sequences
            unique_ids = cv_df['unique_id'].unique()[:max_sequences]
            action.log(message_type="plotting_sequences", count=len(unique_ids))
            
            for seq_id in unique_ids:
                try:
                    # Filter data for this sequence
                    df_seq = df_pandas[df_pandas['unique_id'] == seq_id]
                    cv_seq = cv_df[cv_df['unique_id'] == seq_id]
                    
                    if len(cv_seq) == 0:
                        continue
                    
                    # Use utilsforecast's plot_series
                    fig = plot_series(
                        df_seq,
                        cv_seq.drop(columns=['cutoff'] if 'cutoff' in cv_seq.columns else []),
                        models=[model_name],
                        level=None,
                        max_insample_length=None,  # Show all historical points
                        plot_random=False,
                        ids=[seq_id],
                        engine="matplotlib"
                    )

                    # Customize the plot for better readability
                    fig.set_size_inches(16, 6)  # Make plot wider and shorter for more compact layout

                    # Rotate x-axis labels vertically for better space utilization
                    for ax in fig.get_axes():
                        ax.tick_params(axis='x', labelrotation=90, labelsize=8)
                        # Make x-axis labels more compact
                        for label in ax.get_xticklabels():
                            label.set_horizontalalignment('center')

                    # Save plot
                    plot_file = plots_dir / f'sequence_{seq_id}.png'
                    fig.savefig(str(plot_file), dpi=150, bbox_inches='tight')
                    action.log(message_type="plot_saved", sequence=seq_id, file=str(plot_file))
                    
                except Exception as e:
                    action.log(message_type="plot_warning", sequence=seq_id, error=str(e))
                    continue
                    
        except Exception as e:
            action.log(message_type="plotting_error", error=str(e))

