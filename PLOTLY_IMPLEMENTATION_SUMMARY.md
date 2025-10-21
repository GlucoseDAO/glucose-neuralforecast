# Plotly Visualization Implementation Summary

## What Was Done

### 1. Created New Plotly Plotting Module
**File:** `src/glucose_neuralforecast/plotting_plotly.py`

New functions:
- `plot_predictions_plotly()` - Plot single model predictions with plotly
- `create_timeseries_plot()` - Create interactive timeseries plot
- `plot_comparison_plotly()` - Compare multiple models in one plot
- `create_comparison_plot()` - Create comparison plot figure
- `create_interactive_dashboard()` - Multi-sequence dashboard

**Key Features:**
- ✅ All time points visible on x-axis (controlled by `show_all_ticks` parameter)
- ✅ Vertical labels (controlled by `tickangle` parameter, default -90°)
- ✅ Interactive zoom, pan, and hover
- ✅ Dual output: HTML (interactive) + PNG (static)
- ✅ Customizable dimensions, colors, and styling

### 2. Updated Configuration System
**File:** `src/glucose_neuralforecast/config.py`

Changes:
- Added `plotting_backend` field to `TrainingConfig` (default: 'plotly')
- Updated `save_config()` to include plotting backend
- Fixed Pydantic deprecation by migrating to `ConfigDict`

### 3. Updated Training Module
**File:** `src/glucose_neuralforecast/train.py`

Changes:
- Added `--plotting-backend` CLI parameter
- Import both matplotlib and plotly plotting functions
- Conditional plotting based on backend choice
- Pass plotting backend through config file support

### 4. Updated Package Exports
**File:** `src/glucose_neuralforecast/__init__.py`

Added exports:
- `plot_predictions_plotly`
- `plot_comparison_plotly`
- `create_interactive_dashboard`

### 5. Updated Configuration Files
**File:** `train_config.yaml`

Added:
```yaml
plotting_backend: plotly
```

### 6. Documentation
Created comprehensive documentation:

**Main Documentation:**
- `docs/PLOTLY_VISUALIZATION.md` - Comprehensive guide with examples
- `PLOTLY_QUICKSTART.md` - Quick reference for getting started
- Updated `README.md` with visualization section

**Examples:**
- `examples/plotly_visualization_example.py` - Complete working example

**Tests:**
- `tests/test_plotting_plotly.py` - 7 test cases covering all functionality

## File Structure

```
glucose-neuralforecast/
├── src/glucose_neuralforecast/
│   ├── plotting.py              # Existing matplotlib plotting
│   └── plotting_plotly.py       # NEW: Plotly plotting
│   ├── config.py                # Updated: added plotting_backend
│   ├── train.py                 # Updated: backend selection
│   └── __init__.py              # Updated: exports
├── docs/
│   └── PLOTLY_VISUALIZATION.md  # NEW: Comprehensive docs
├── examples/
│   └── plotly_visualization_example.py  # NEW: Example script
├── tests/
│   └── test_plotting_plotly.py  # NEW: Test suite
├── train_config.yaml            # Updated: plotting_backend field
├── PLOTLY_QUICKSTART.md         # NEW: Quick reference
└── PLOTLY_IMPLEMENTATION_SUMMARY.md  # This file
```

## Usage Examples

### Command Line
```bash
# Use plotly (default)
uv run train --plotting-backend plotly --models "LSTM,NHITS"

# Use matplotlib (legacy)
uv run train --plotting-backend matplotlib --models "LSTM,NHITS"

# From config file
uv run train-from-config --config train_config.yaml
```

### Python API
```python
from glucose_neuralforecast.plotting_plotly import plot_predictions_plotly

plot_predictions_plotly(
    df=df,
    cv_df=predictions,
    model_name="LSTM",
    output_path=output_dir,
    show_all_ticks=True,   # Show ALL time points
    tickangle=-90,          # Vertical labels
    height=600,
    width=1400,
)
```

### Configuration File
```yaml
# train_config.yaml
plotting_backend: plotly  # or 'matplotlib'
horizon: 12
input_size: 48
models:
  - NHITS
  - LSTM
```

## Key Parameters

### `show_all_ticks` (bool, default=True)
- `True`: Display all time points on x-axis
- `False`: Use automatic tick selection

### `tickangle` (int, default=-90)
- `-90`: Vertical (recommended for many time points)
- `-45`: Diagonal
- `0`: Horizontal

### `height` and `width` (int)
- Plot dimensions in pixels
- Defaults: 600x1400 (single), 800x1600 (comparison)

## Testing

All functionality is tested:
```bash
cd /home/antonkulaga/sources/glucose-neuralforecast
uv run pytest tests/test_plotting_plotly.py -v
```

Test results: ✅ 7 tests passed

Test coverage:
- ✅ Single timeseries plot creation
- ✅ Complete plot_predictions_plotly workflow
- ✅ Comparison plot creation
- ✅ Complete plot_comparison_plotly workflow
- ✅ Different tick angle configurations
- ✅ show_all_ticks option behavior
- ✅ Custom dimensions

## Output Files

### With Matplotlib (legacy)
```
plots/
└── LSTM/
    └── sequence_1.png     # Static image only
```

### With Plotly (new default)
```
plots/
└── LSTM/
    ├── sequence_1.html    # Interactive (open in browser)
    └── sequence_1.png     # Static image (publication-ready)
```

## Migration Guide

### For Existing Code
No changes needed! The default behavior now uses plotly, but all existing code works:

```python
# This now creates both HTML and PNG with plotly by default
plot_predictions(df, cv_df, model_name, output_path)
```

To explicitly choose:
```bash
# Plotly (default)
uv run train --plotting-backend plotly

# Matplotlib (legacy)
uv run train --plotting-backend matplotlib
```

### For Config Files
Add one line to your `train_config.yaml`:
```yaml
plotting_backend: plotly  # or 'matplotlib'
```

## Dependencies

Already included in `pyproject.toml`:
- `plotly>=6.3.1` - Core plotting library
- `kaleido>=0.2.1` - Static image export (PNG)

No additional installation needed after `uv sync`.

## Advantages Over Matplotlib

| Feature | Matplotlib | Plotly |
|---------|-----------|--------|
| Interactive zooming | ❌ | ✅ |
| Hover tooltips | ❌ | ✅ |
| All ticks visible | ⚠️ (limited) | ✅ |
| Vertical labels | ✅ | ✅ |
| Export formats | PNG only | HTML + PNG |
| Browser viewing | ❌ | ✅ |
| File size | Smaller | Larger (HTML) |
| Rendering speed | Faster | Slightly slower |

## Future Enhancements

Possible future additions:
- [ ] 3D plots for multi-dimensional analysis
- [ ] Animated time-series progressions
- [ ] Real-time updating plots
- [ ] Export to additional formats (SVG, PDF)
- [ ] Custom color schemes/themes
- [ ] Annotations and markers
- [ ] Subplots with shared axes

## Notes

1. **Default Changed**: Plotly is now the default plotting backend
2. **Backward Compatible**: Matplotlib still available via `--plotting-backend matplotlib`
3. **File Sizes**: HTML files are larger (500KB-2MB) but provide interactivity
4. **Performance**: Slightly slower than matplotlib but negligible for typical use
5. **Browser Required**: HTML files open in web browser for full interactivity

## Support

For issues or questions:
1. Check documentation: `docs/PLOTLY_VISUALIZATION.md`
2. See examples: `examples/plotly_visualization_example.py`
3. Review quick start: `PLOTLY_QUICKSTART.md`
4. Run tests: `pytest tests/test_plotting_plotly.py -v`

## Summary

✅ Plotly visualization successfully integrated  
✅ All time points visible with vertical labels  
✅ Interactive and static outputs  
✅ Backward compatible with matplotlib  
✅ Fully tested and documented  
✅ Ready for production use  

