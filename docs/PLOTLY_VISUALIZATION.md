# Plotly-Based Visualization

This project now includes enhanced plotly-based visualization capabilities with improved tick display and interactivity.

## Features

### 1. **All Time Points Visible**
- Unlike matplotlib plots that may skip ticks, plotly plots show ALL time points
- No data points are hidden or omitted from the x-axis

### 2. **Vertical Labels for Space Efficiency**
- Tick labels are rotated vertically (-90 degrees) to save horizontal space
- Prevents label overlapping even with many time points

### 3. **Interactive Plots**
- Zoom in/out with mouse wheel or selection
- Pan by clicking and dragging
- Hover over points to see exact values
- Toggle series on/off by clicking legend items
- Double-click to reset view

### 4. **Dual Output Formats**
- **HTML**: Fully interactive plots that can be opened in any browser
- **PNG**: Static publication-ready images for reports

## Usage

### Command Line

#### Using the `--plotting-backend` flag:

```bash
# Train models with plotly visualization
train --plotting-backend plotly --models "LSTM,NHITS" --max-steps 1000

# Or use matplotlib (legacy)
train --plotting-backend matplotlib --models "LSTM,NHITS" --max-steps 1000
```

#### Using configuration file:

```yaml
# train_config.yaml
plotting_backend: plotly  # or 'matplotlib'
horizon: 12
input_size: 48
max_steps: 1000
models:
  - NHITS
  - LSTM
  - TCN
```

```bash
train-from-config --config train_config.yaml
```

### Python API

#### Simple Plotting

```python
from glucose_neuralforecast.plotting_plotly import plot_predictions_plotly
from glucose_neuralforecast.data import load_glucose_data

# Load data
df = load_glucose_data('data/input/livia_glucose.csv')

# After getting predictions (cv_df)...
plot_predictions_plotly(
    df=df,
    cv_df=cv_df,
    model_name="LSTM",
    output_path=Path("output/plots"),
    max_sequences=3,
    show_all_ticks=True,  # Show every time point
    tickangle=-90,         # Vertical labels
    height=600,
    width=1400,
)
```

#### Model Comparison

```python
from glucose_neuralforecast.plotting_plotly import plot_comparison_plotly

# Compare multiple models for a single sequence
plot_comparison_plotly(
    df=df,
    predictions={
        'LSTM': lstm_predictions,
        'NHITS': nhits_predictions,
        'TCN': tcn_predictions,
    },
    output_path=Path("output/plots"),
    sequence_id="seq_1",
    show_all_ticks=True,
    tickangle=-90,
)
```

#### Interactive Dashboard

```python
from glucose_neuralforecast.plotting_plotly import create_interactive_dashboard

# Create a dashboard with multiple sequences and models
create_interactive_dashboard(
    df=df,
    predictions=predictions_dict,
    metrics=metrics_df,
    output_path=Path("output/plots"),
    max_sequences=5,
)
```

## Configuration Options

### `show_all_ticks` (bool, default=True)
- `True`: Display all time points on x-axis
- `False`: Use automatic tick selection (fewer ticks)

### `tickangle` (int, default=-90)
- Angle in degrees for tick labels
- `-90`: Vertical (recommended for many time points)
- `-45`: Diagonal
- `0`: Horizontal

### `height` and `width` (int)
- Plot dimensions in pixels
- Default: 600x1400 for single plots
- Default: 800x1600 for comparison plots

## Output Files

When using plotly, two files are created for each plot:

1. **HTML file**: `sequence_X.html` or `comparison_X.html`
   - Fully interactive
   - Can be opened in any web browser
   - Supports zooming, panning, and tooltips

2. **PNG file**: `sequence_X.png` or `comparison_X.png`
   - Static image
   - Publication-ready
   - Can be embedded in documents

## Examples

See the complete examples in:
- `examples/plotly_visualization_example.py`

Run the example:
```bash
cd /home/antonkulaga/sources/glucose-neuralforecast
python examples/plotly_visualization_example.py
```

## Advantages over Matplotlib

| Feature | Matplotlib | Plotly |
|---------|-----------|--------|
| Interactive zooming | ❌ | ✅ |
| Hover tooltips | ❌ | ✅ |
| All ticks visible | ⚠️ (limited) | ✅ |
| Vertical labels | ✅ | ✅ |
| Export formats | PNG only | HTML + PNG |
| File size | Smaller | Larger (HTML) |
| Speed | Faster | Slightly slower |

## When to Use Which Backend

### Use **Plotly** (default) when:
- You need interactive exploration of results
- You want to see ALL time points clearly
- You're creating reports or presentations
- You need to zoom in on specific time ranges
- You want hover information for exact values

### Use **Matplotlib** when:
- You only need static images
- File size is a concern (many plots)
- You prefer the matplotlib aesthetic
- You're running in a constrained environment

## Technical Details

### Dependencies
- `plotly>=6.3.1`: Core plotting library
- `kaleido>=0.2.1`: Static image export (PNG)

Both are already included in the project dependencies.

### Plot Styling
- Color palette: Carefully selected colors for model differentiation
- Font: Arial, sans-serif
- Background: White (both plot and paper)
- Grid: Light gray with transparency
- Legend: Horizontal layout for single plots, vertical for comparisons

### Performance
- Initial rendering: Slightly slower than matplotlib
- File size: HTML files are larger (~500KB - 2MB depending on data)
- PNG export: Uses kaleido for high-quality static images
- Interaction: Smooth even with many data points

## Troubleshooting

### "kaleido not found" error
Install kaleido for PNG export:
```bash
uv add kaleido
```

### Plots are slow to load
- Reduce `max_sequences` parameter
- Consider using `show_all_ticks=False` for very large datasets

### Labels are overlapping
- Increase `tickangle` (make more vertical)
- Increase plot width
- Use `show_all_ticks=False`

## Migration from Matplotlib

Your existing code will continue to work with matplotlib by default. To switch to plotly:

1. Update your config file to add `plotting_backend: plotly`
2. Or pass `--plotting-backend plotly` to command line
3. No code changes needed!

All plots will now be generated with plotly, creating both HTML and PNG outputs.

