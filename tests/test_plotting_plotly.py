"""Tests for plotly plotting functionality."""

import pytest
from pathlib import Path
import polars as pl
import pandas as pd
import tempfile

from glucose_neuralforecast.plotting_plotly import (
    plot_predictions_plotly,
    create_timeseries_plot,
    plot_comparison_plotly,
    create_comparison_plot,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample timeseries data
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    
    df = pl.DataFrame({
        'unique_id': ['seq_1'] * 100,
        'ds': dates,
        'y': range(100, 200),
    })
    
    # Create sample predictions
    pred_dates = dates[-20:]  # Last 20 points are predictions
    cv_df = pd.DataFrame({
        'unique_id': ['seq_1'] * 20,
        'ds': pred_dates,
        'y': range(180, 200),
        'TestModel': range(178, 198),  # Slightly different predictions
    })
    
    return df, cv_df


def test_create_timeseries_plot(sample_data):
    """Test creating a single timeseries plot."""
    df, cv_df = sample_data
    df_pandas = df.to_pandas()
    
    fig = create_timeseries_plot(
        df_seq=df_pandas,
        cv_seq=cv_df,
        model_name='TestModel',
        seq_id='seq_1',
        show_all_ticks=True,
        tickangle=-90,
    )
    
    # Check that figure was created
    assert fig is not None
    assert len(fig.data) == 2  # Actual and predicted traces
    
    # Check trace names
    trace_names = [trace.name for trace in fig.data]
    assert 'Actual' in trace_names
    assert 'Predicted' in trace_names


def test_plot_predictions_plotly(sample_data, tmp_path):
    """Test the complete plot_predictions_plotly function."""
    df, cv_df = sample_data
    
    plot_predictions_plotly(
        df=df,
        cv_df=cv_df,
        model_name='TestModel',
        output_path=tmp_path,
        max_sequences=1,
        show_all_ticks=True,
        tickangle=-90,
    )
    
    # Check that output files were created
    plots_dir = tmp_path / 'plots' / 'TestModel'
    assert plots_dir.exists()
    
    html_file = plots_dir / 'sequence_seq_1.html'
    png_file = plots_dir / 'sequence_seq_1.png'
    
    assert html_file.exists()
    assert png_file.exists()
    
    # Check file sizes (basic validation)
    assert html_file.stat().st_size > 0
    assert png_file.stat().st_size > 0


def test_create_comparison_plot(sample_data):
    """Test creating a comparison plot with multiple models."""
    df, cv_df = sample_data
    df_pandas = df.to_pandas()
    
    # Create predictions for multiple models
    predictions = {
        'Model1': cv_df.copy(),
        'Model2': cv_df.copy(),
    }
    predictions['Model2']['Model2'] = cv_df['TestModel'] + 2
    
    fig = create_comparison_plot(
        df_seq=df_pandas,
        predictions=predictions,
        sequence_id='seq_1',
        show_all_ticks=True,
        tickangle=-90,
    )
    
    # Check that figure was created
    assert fig is not None
    # Should have: 1 actual trace + 2 model traces = 3 total
    assert len(fig.data) >= 2
    
    # Check that actual trace exists
    trace_names = [trace.name for trace in fig.data]
    assert 'Actual' in trace_names


def test_plot_comparison_plotly(sample_data, tmp_path):
    """Test the complete plot_comparison_plotly function."""
    df, cv_df = sample_data
    
    # Create predictions for multiple models
    predictions = {
        'Model1': cv_df.copy(),
        'Model2': cv_df.copy(),
    }
    predictions['Model1']['Model1'] = cv_df['TestModel']
    predictions['Model2']['Model2'] = cv_df['TestModel'] + 2
    
    plot_comparison_plotly(
        df=df,
        predictions=predictions,
        output_path=tmp_path,
        sequence_id='seq_1',
        show_all_ticks=True,
        tickangle=-90,
    )
    
    # Check that output files were created
    plots_dir = tmp_path / 'plots' / 'comparison'
    assert plots_dir.exists()
    
    html_file = plots_dir / 'comparison_seq_1.html'
    png_file = plots_dir / 'comparison_seq_1.png'
    
    assert html_file.exists()
    assert png_file.exists()


def test_tick_angle_variations(sample_data):
    """Test different tick angle configurations."""
    df, cv_df = sample_data
    df_pandas = df.to_pandas()
    
    for angle in [0, -45, -90]:
        fig = create_timeseries_plot(
            df_seq=df_pandas,
            cv_seq=cv_df,
            model_name='TestModel',
            seq_id='seq_1',
            tickangle=angle,
        )
        
        assert fig is not None
        # Check that tickangle is applied
        assert fig.layout.xaxis.tickangle == angle


def test_show_all_ticks_option(sample_data):
    """Test the show_all_ticks option."""
    df, cv_df = sample_data
    df_pandas = df.to_pandas()
    
    # Test with show_all_ticks=True
    fig_all = create_timeseries_plot(
        df_seq=df_pandas,
        cv_seq=cv_df,
        model_name='TestModel',
        seq_id='seq_1',
        show_all_ticks=True,
    )
    
    # Test with show_all_ticks=False
    fig_auto = create_timeseries_plot(
        df_seq=df_pandas,
        cv_seq=cv_df,
        model_name='TestModel',
        seq_id='seq_1',
        show_all_ticks=False,
    )
    
    assert fig_all is not None
    assert fig_auto is not None
    
    # With show_all_ticks=True, should have explicit tickvals
    assert fig_all.layout.xaxis.tickmode == 'array'
    
    # With show_all_ticks=False, should not have tickmode='array'
    assert fig_auto.layout.xaxis.tickmode != 'array'


def test_custom_dimensions(sample_data, tmp_path):
    """Test custom plot dimensions."""
    df, cv_df = sample_data
    
    custom_height = 800
    custom_width = 2000
    
    plot_predictions_plotly(
        df=df,
        cv_df=cv_df,
        model_name='TestModel',
        output_path=tmp_path,
        max_sequences=1,
        height=custom_height,
        width=custom_width,
    )
    
    # Verify files were created
    plots_dir = tmp_path / 'plots' / 'TestModel'
    assert (plots_dir / 'sequence_seq_1.html').exists()
    assert (plots_dir / 'sequence_seq_1.png').exists()


