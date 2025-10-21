# Plotly Visualization Quick Start

## What Changed?

Your plotting utility now has **plotly support** with enhanced features:

✅ **All time points visible** - no more skipped ticks  
✅ **Vertical labels** - saves space, no overlapping  
✅ **Interactive plots** - zoom, pan, hover for exact values  
✅ **Dual formats** - HTML (interactive) + PNG (static)  

## Quick Usage

### 1. Command Line (Easiest)

```bash
# Use plotly (default now)
uv run train --plotting-backend plotly --models "LSTM,NHITS" --max-steps 1000

# Or explicitly use matplotlib (legacy)
uv run train --plotting-backend matplotlib --models "LSTM,NHITS" --max-steps 1000
```

### 2. Config File

Add to your `train_config.yaml`:

```yaml
plotting_backend: plotly  # or 'matplotlib'
```

Then run:

```bash
uv run train-from-config --config train_config.yaml
```

### 3. Python API

```python
from glucose_neuralforecast.plotting_plotly import plot_predictions_plotly

# After training/prediction...
plot_predictions_plotly(
    df=df,
    cv_df=predictions,
    model_name="LSTM",
    output_path=output_dir,
    show_all_ticks=True,  # Show ALL time points
    tickangle=-90,         # Vertical labels
)
```

## What You Get

### With Matplotlib (legacy):
- `plots/LSTM/sequence_1.png` - Static image only

### With Plotly (new default):
- `plots/LSTM/sequence_1.html` - Interactive (open in browser!)
- `plots/LSTM/sequence_1.png` - Static image (same as before)

## Key Parameters

```python
plot_predictions_plotly(
    show_all_ticks=True,   # True = show every tick, False = auto select
    tickangle=-90,         # -90 = vertical, -45 = diagonal, 0 = horizontal
    height=600,            # Plot height in pixels
    width=1400,            # Plot width in pixels
)
```

## Examples

See full examples in:
- `examples/plotly_visualization_example.py`
- `docs/PLOTLY_VISUALIZATION.md`

## Try It Now

```bash
# Quick test with existing run
cd /home/antonkulaga/sources/glucose-neuralforecast
uv run python examples/plotly_visualization_example.py
```

## Need Help?

See the detailed guide: [docs/PLOTLY_VISUALIZATION.md](docs/PLOTLY_VISUALIZATION.md)


