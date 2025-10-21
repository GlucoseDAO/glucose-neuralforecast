# Cherry-Pick Inference Results - Run 20251021_131309

## Summary
Successfully ran cherry-pick inference on the last training run using all 19 trained models.

### Inference Details
- **Selected Sequence**: unique_id = 3 (best sequence by mean MAE)
- **Number of Models**: 19
- **Sequence Data Points**: 330 rows
- **Prediction Horizon**: 12 time steps (60 minutes at 5-minute intervals)
- **Data Frequency**: 5 minutes

### Models Used (All Successful)
1. Autoformer
2. BiTCN
3. DeepAR
4. DeepNPTS
5. DilatedRNN
6. FEDformer
7. GRU
8. Informer
9. KAN
10. LSTM
11. MLP
12. NBEATSx
13. NHITS
14. RNN
15. TCN
16. TFT
17. TiDE
18. TimesNet
19. VanillaTransformer

### Output Location
```
/home/antonkulaga/sources/glucose-neuralforecast/data/output/runs/run_20251021_131309/predictions/
```

### Generated Files
- 19 CSV files containing predictions from each model
- Each file contains 12 predictions for sequence 3
- Columns: `unique_id`, `ds` (datetime), `[model_name]` (predicted value)

## Bug Fixes Applied

### Issue 1: Cherry-pick Function Error
**Problem**: The `cherry_pick_sequence()` function was trying to calculate mean across all columns including string columns, causing a type error.

**Solution**: Updated to:
1. Filter to only numeric model columns
2. Handle null values in model performance metrics
3. Calculate mean properly using polars list operations

### Issue 2: Polars Frequency Compatibility
**Problem**: Newer versions of polars don't support the 'min' time unit. Models were trained with '5min' frequency which causes an error during prediction.

**Error**: `unit: 'min' not supported; available units are: 'y', 'mo', 'q', 'w', 'd', 'h', 'm', 's', 'ms', 'us', 'ns'`

**Solution**: Updated `load_trained_model()` to automatically convert 'min' to 'm' in the frequency string:
```python
if hasattr(nf, 'freq') and nf.freq and 'min' in nf.freq:
    nf.freq = nf.freq.replace('min', 'm')
```

This ensures compatibility with the current version of polars while maintaining the correct frequency specification.

## Sample Prediction
```
unique_id,ds,NHITS
3,2024-04-05T04:27:00.000000,39.60334
3,2024-04-05T04:32:00.000000,39.720062
3,2024-04-05T04:37:00.000000,39.800102
3,2024-04-05T04:42:00.000000,39.92416
...
```

## Run Time
- Model inference completed successfully with no errors
- All 19 models generated predictions
- Predictions saved to CSV files

## Command to Reproduce
```bash
cd /home/antonkulaga/sources/glucose-neuralforecast
uv run predict predict --run-id run_20251021_131309 --cherry-pick --best
```

## Additional Options
- Use random sequence: `--cherry-pick --random`
- Use specific sequence: `--unique-id 5`
- Select specific models: `--models "NHITS,LSTM,Autoformer"`
- Generate plots: `--plot` (requires kaleido package)
