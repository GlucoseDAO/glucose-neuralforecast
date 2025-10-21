"""Plotly-based plotting functions for visualizing model predictions."""

from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

import polars as pl
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from eliot import start_action


def plot_predictions_plotly(
    df: pl.DataFrame,
    cv_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
    max_sequences: int = 3,
    show_all_ticks: bool = True,
    tickangle: int = -90,
    height: int = 600,
    width: int = 1400,
) -> None:
    """
    Plot prediction examples for a model using Plotly with enhanced interactivity.
    
    Args:
        df: Original data (polars)
        cv_df: Cross-validation results with predictions (pandas)
        model_name: Name of the model
        output_path: Directory to save plots
        max_sequences: Maximum number of sequences to plot
        show_all_ticks: Whether to show all time point ticks on x-axis
        tickangle: Angle for tick labels (negative = counterclockwise, -90 = vertical)
        height: Plot height in pixels
        width: Plot width in pixels
    """
    plots_dir = output_path / 'plots' / model_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    with start_action(action_type="plot_predictions_plotly", model=model_name) as action:
        try:
            # Convert to pandas if needed
            df_pandas = df.to_pandas() if isinstance(df, pl.DataFrame) else df
            
            # Get unique sequences
            unique_ids = cv_df['unique_id'].unique()[:max_sequences]
            action.log(message_type="plotting_sequences", count=len(unique_ids))
            
            for seq_id in unique_ids:
                try:
                    # Filter data for this sequence
                    df_seq = df_pandas[df_pandas['unique_id'] == seq_id].copy()
                    cv_seq = cv_df[cv_df['unique_id'] == seq_id].copy()
                    
                    if len(cv_seq) == 0:
                        continue
                    
                    # Create the plot
                    fig = create_timeseries_plot(
                        df_seq=df_seq,
                        cv_seq=cv_seq,
                        model_name=model_name,
                        seq_id=seq_id,
                        show_all_ticks=show_all_ticks,
                        tickangle=tickangle,
                        height=height,
                        width=width,
                    )
                    
                    # Save as interactive HTML
                    html_file = plots_dir / f'sequence_{seq_id}.html'
                    fig.write_html(str(html_file))
                    action.log(message_type="plot_saved", sequence=seq_id, file=str(html_file), format="html")
                    
                    # Also save as static PNG using kaleido
                    png_file = plots_dir / f'sequence_{seq_id}.png'
                    fig.write_image(str(png_file), width=width, height=height)
                    action.log(message_type="plot_saved", sequence=seq_id, file=str(png_file), format="png")
                    
                except Exception as e:
                    action.log(message_type="plot_warning", sequence=seq_id, error=str(e))
                    continue
                    
        except Exception as e:
            action.log(message_type="plotting_error", error=str(e))


def create_timeseries_plot(
    df_seq: pd.DataFrame,
    cv_seq: pd.DataFrame,
    model_name: str,
    seq_id: Any,
    show_all_ticks: bool = True,
    tickangle: int = -90,
    height: int = 600,
    width: int = 1400,
) -> go.Figure:
    """
    Create an interactive plotly timeseries plot for a single sequence.
    
    Args:
        df_seq: Original data for the sequence
        cv_seq: Cross-validation results for the sequence
        model_name: Name of the model
        seq_id: Unique identifier for the sequence
        show_all_ticks: Whether to show all time point ticks on x-axis
        tickangle: Angle for tick labels (negative = counterclockwise)
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Add actual values trace
    fig.add_trace(go.Scatter(
        x=df_seq['ds'],
        y=df_seq['y'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=4, symbol='circle'),
        hovertemplate='<b>Actual</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>',
    ))
    
    # Add predicted values trace
    if model_name in cv_seq.columns:
        fig.add_trace(go.Scatter(
            x=cv_seq['ds'],
            y=cv_seq[model_name],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#F24236', width=2, dash='dash'),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate='<b>Predicted</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>',
        ))
    
    # Configure x-axis with all ticks if requested
    xaxis_config = dict(
        title='Time',
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.2)',
    )
    
    if show_all_ticks:
        # Show all time points
        xaxis_config.update(dict(
            tickmode='array',
            tickvals=df_seq['ds'].tolist(),
            ticktext=[str(d) for d in df_seq['ds'].tolist()],
            tickangle=tickangle,
            tickfont=dict(size=8),
        ))
    else:
        # Use auto ticks
        xaxis_config.update(dict(
            tickangle=tickangle,
            tickfont=dict(size=10),
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{model_name} - Sequence {seq_id}',
            font=dict(size=20, family='Arial, sans-serif'),
            x=0.5,
            xanchor='center',
        ),
        xaxis=xaxis_config,
        yaxis=dict(
            title='Value',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=12),
        ),
        height=height,
        width=width,
        margin=dict(l=80, r=50, t=100, b=150),  # Extra bottom margin for vertical labels
    )
    
    return fig


def plot_comparison_plotly(
    df: pl.DataFrame,
    predictions: Dict[str, pd.DataFrame],
    output_path: Path,
    sequence_id: str,
    models: Optional[List[str]] = None,
    show_all_ticks: bool = True,
    tickangle: int = -90,
    height: int = 800,
    width: int = 1600,
) -> None:
    """
    Create a comparison plot of multiple models for a single sequence using Plotly.
    
    Args:
        df: Original data (polars)
        predictions: Dict mapping model names to their prediction DataFrames
        output_path: Directory to save plots
        sequence_id: ID of the sequence to plot
        models: List of model names to include (None = all)
        show_all_ticks: Whether to show all time point ticks on x-axis
        tickangle: Angle for tick labels
        height: Plot height in pixels
        width: Plot width in pixels
    """
    plots_dir = output_path / 'plots' / 'comparison'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    with start_action(action_type="plot_comparison_plotly", sequence=sequence_id) as action:
        try:
            # Convert to pandas if needed
            df_pandas = df.to_pandas() if isinstance(df, pl.DataFrame) else df
            df_seq = df_pandas[df_pandas['unique_id'] == sequence_id].copy()
            
            if len(df_seq) == 0:
                action.log(message_type="sequence_not_found", sequence=sequence_id)
                return
            
            # Filter models if specified
            if models is not None:
                predictions = {k: v for k, v in predictions.items() if k in models}
            
            fig = create_comparison_plot(
                df_seq=df_seq,
                predictions=predictions,
                sequence_id=sequence_id,
                show_all_ticks=show_all_ticks,
                tickangle=tickangle,
                height=height,
                width=width,
            )
            
            # Save as interactive HTML
            html_file = plots_dir / f'comparison_{sequence_id}.html'
            fig.write_html(str(html_file))
            action.log(message_type="comparison_saved", sequence=sequence_id, file=str(html_file), format="html")
            
            # Save as static PNG
            png_file = plots_dir / f'comparison_{sequence_id}.png'
            fig.write_image(str(png_file), width=width, height=height)
            action.log(message_type="comparison_saved", sequence=sequence_id, file=str(png_file), format="png")
            
        except Exception as e:
            action.log(message_type="comparison_error", sequence=sequence_id, error=str(e))


def create_comparison_plot(
    df_seq: pd.DataFrame,
    predictions: Dict[str, pd.DataFrame],
    sequence_id: str,
    show_all_ticks: bool = True,
    tickangle: int = -90,
    height: int = 800,
    width: int = 1600,
) -> go.Figure:
    """
    Create an interactive plotly comparison plot for multiple models.
    
    Args:
        df_seq: Original data for the sequence
        predictions: Dict mapping model names to prediction DataFrames
        sequence_id: ID of the sequence
        show_all_ticks: Whether to show all time point ticks
        tickangle: Angle for tick labels
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Color palette for different models
    colors = [
        '#2E86AB', '#F24236', '#F6AE2D', '#55A630', '#9D4EDD',
        '#E63946', '#06FFA5', '#457B9D', '#F77F00', '#D62828',
        '#023047', '#FB5607', '#8338EC', '#3A86FF', '#FFBE0B',
    ]
    
    # Add actual values trace
    fig.add_trace(go.Scatter(
        x=df_seq['ds'],
        y=df_seq['y'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='black', width=3),
        marker=dict(size=5, symbol='circle'),
        hovertemplate='<b>Actual</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>',
    ))
    
    # Add prediction traces for each model
    for idx, (model_name, pred_df) in enumerate(predictions.items()):
        pred_seq = pred_df[pred_df['unique_id'] == sequence_id].copy()
        
        if len(pred_seq) == 0 or model_name not in pred_seq.columns:
            continue
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=pred_seq['ds'],
            y=pred_seq[model_name],
            mode='lines+markers',
            name=model_name,
            line=dict(color=color, width=2, dash='dash'),
            marker=dict(size=4, symbol='diamond'),
            hovertemplate=f'<b>{model_name}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>',
        ))
    
    # Configure x-axis
    xaxis_config = dict(
        title='Time',
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.2)',
    )
    
    if show_all_ticks:
        xaxis_config.update(dict(
            tickmode='array',
            tickvals=df_seq['ds'].tolist(),
            ticktext=[str(d) for d in df_seq['ds'].tolist()],
            tickangle=tickangle,
            tickfont=dict(size=8),
        ))
    else:
        xaxis_config.update(dict(
            tickangle=tickangle,
            tickfont=dict(size=10),
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Model Comparison - Sequence {sequence_id}',
            font=dict(size=22, family='Arial, sans-serif'),
            x=0.5,
            xanchor='center',
        ),
        xaxis=xaxis_config,
        yaxis=dict(
            title='Value',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.01,
            font=dict(size=10),
        ),
        height=height,
        width=width,
        margin=dict(l=80, r=150, t=100, b=150),
    )
    
    return fig


def create_interactive_dashboard(
    df: pl.DataFrame,
    predictions: Dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    output_path: Path,
    sequence_ids: Optional[List[str]] = None,
    max_sequences: int = 5,
) -> None:
    """
    Create an interactive dashboard with multiple subplots.
    
    Args:
        df: Original data (polars)
        predictions: Dict mapping model names to prediction DataFrames
        metrics: Metrics DataFrame with model performance
        output_path: Directory to save the dashboard
        sequence_ids: Specific sequence IDs to include (None = random selection)
        max_sequences: Maximum number of sequences to include
    """
    plots_dir = output_path / 'plots' / 'dashboard'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    with start_action(action_type="create_interactive_dashboard") as action:
        try:
            df_pandas = df.to_pandas() if isinstance(df, pl.DataFrame) else df
            
            # Select sequences
            if sequence_ids is None:
                all_ids = df_pandas['unique_id'].unique()
                import random
                sequence_ids = list(all_ids[:max_sequences])
            
            # Create subplots - one row per sequence
            n_sequences = len(sequence_ids)
            fig = make_subplots(
                rows=n_sequences,
                cols=1,
                subplot_titles=[f'Sequence {sid}' for sid in sequence_ids],
                vertical_spacing=0.1,
            )
            
            colors = ['#2E86AB', '#F24236', '#F6AE2D', '#55A630', '#9D4EDD']
            
            for row_idx, seq_id in enumerate(sequence_ids, start=1):
                df_seq = df_pandas[df_pandas['unique_id'] == seq_id]
                
                # Add actual trace
                fig.add_trace(
                    go.Scatter(
                        x=df_seq['ds'],
                        y=df_seq['y'],
                        mode='lines+markers',
                        name=f'Actual ({seq_id})',
                        line=dict(color='black', width=2),
                        marker=dict(size=4),
                        showlegend=(row_idx == 1),
                    ),
                    row=row_idx, col=1
                )
                
                # Add predictions for each model
                for model_idx, (model_name, pred_df) in enumerate(predictions.items()):
                    pred_seq = pred_df[pred_df['unique_id'] == seq_id]
                    
                    if len(pred_seq) > 0 and model_name in pred_seq.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=pred_seq['ds'],
                                y=pred_seq[model_name],
                                mode='lines+markers',
                                name=f'{model_name}',
                                line=dict(color=colors[model_idx % len(colors)], width=2, dash='dash'),
                                marker=dict(size=3),
                                showlegend=(row_idx == 1),
                            ),
                            row=row_idx, col=1
                        )
            
            # Update layout
            fig.update_layout(
                height=300 * n_sequences,
                width=1600,
                title_text='Multi-Sequence Dashboard',
                showlegend=True,
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white',
            )
            
            # Update all x-axes to show vertical labels
            for i in range(1, n_sequences + 1):
                fig.update_xaxes(tickangle=-90, row=i, col=1)
                fig.update_yaxes(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', row=i, col=1)
            
            # Save dashboard
            html_file = plots_dir / 'dashboard.html'
            fig.write_html(str(html_file))
            action.log(message_type="dashboard_saved", file=str(html_file), sequences=len(sequence_ids))
            
        except Exception as e:
            action.log(message_type="dashboard_error", error=str(e))

