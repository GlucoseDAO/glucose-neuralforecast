# Windows Setup Guide for glucose-neuralforecast

This guide covers Windows-specific setup and configuration for running glucose-neuralforecast with Ray support.

## Prerequisites

### System Requirements

- **Windows 10/11** (64-bit)
- **Python 3.12 or later**
- **uv** package manager
- At least **8GB RAM** (16GB recommended for larger datasets)
- **Visual C++ Redistributable** (usually installed with Python)

### Python Environment

**Supported Python Versions**: Python 3.12 or later

Ensure you have Python installed from [python.org](https://www.python.org/downloads/) or via Windows Store.

## Installation

### 1. Install with uv

The project uses `uv` for dependency management:

```powershell
# Install uv if you haven't already
pip install uv

# Clone the repository
git clone https://github.com/your-org/glucose-neuralforecast.git
cd glucose-neuralforecast

# Create virtual environment and install dependencies
uv sync
```

### 2. Ray Installation

Ray is installed automatically as a dependency of `neuralforecast`. However, on Windows, there are some specific considerations:

According to the [Ray documentation](https://docs.ray.io/en/latest/ray-overview/installation.html), Windows support is available but with some limitations:

- Ray dashboard may have issues on Windows
- Some Ray features work better in WSL2 (Windows Subsystem for Linux)
- Plasma object store has different behavior on Windows

#### Option A: Standard Installation (Recommended First)

Try the standard installation first:

```powershell
uv sync
```

This will install Ray via neuralforecast's dependencies and should work for most cases.

#### Option B: Windows-Specific Ray Wheel (If Issues Occur)

If you encounter issues with Ray on Windows (especially with Python 3.12), install the Windows-specific Ray wheel first:

```powershell
# For Python 3.12 on Windows (64-bit)
pip install https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp312-cp312-win_amd64.whl

# Then install other dependencies
uv sync
```

**Note**: This uses the development version of Ray (3.0.0.dev0) which may have better Windows compatibility for Python 3.12. The wheel URL is:
- [ray-3.0.0.dev0-cp312-cp312-win_amd64.whl](https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp312-cp312-win_amd64.whl)

**Python Version Compatibility**:
- The project requires Python 3.12 or later
- The Windows wheel above is specifically for Python 3.12 (cp312)
- For other Python versions, check the [Ray wheels repository](https://docs.ray.io/en/latest/ray-overview/installation.html)

#### Option C: Alternative Python Versions

If you're using a different Python version, you can find wheels for:
- Python 3.12: `cp312-cp312-win_amd64.whl`
- Python 3.13: Check Ray's latest wheels for cp313

For the latest Windows-specific Ray wheels, visit the [Ray installation documentation](https://docs.ray.io/en/latest/ray-overview/installation.html#windows-support).

## Configuration

### Ray Configuration

The project includes a `.rayconfig` file with Windows-optimized settings:

```ini
[ray]
include_dashboard = false
object_store_memory = 2000000000  # 2GB

[logging]
log_to_driver = true

[worker]
worker_timeout_seconds = 60
```

### Environment Variables

For better Ray performance on Windows, you can set these environment variables:

```powershell
# In PowerShell
$env:RAY_TMPDIR = "$HOME\ray_tmp"
$env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER = "1"
$env:RAY_DEDUP_LOGS = "0"
```

Or in Command Prompt:

```cmd
set RAY_TMPDIR=%USERPROFILE%\ray_tmp
set RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
set RAY_DEDUP_LOGS=0
```

### Persistent Environment Variables

To make these permanent:

1. Open **System Properties** → **Advanced** → **Environment Variables**
2. Add the variables under **User variables**
3. Restart your terminal

## Ray Initialization in Code

The project provides a `ray_init.py` module with Windows-specific initialization:

```python
from ray_init import configure_ray_environment, init_ray_for_training

# Configure environment before importing neuralforecast
configure_ray_environment()

# Optional: manually initialize Ray with specific settings
init_ray_for_training(num_cpus=4, verbose=True)
```

However, `neuralforecast` typically handles Ray initialization internally, so manual initialization is often not necessary.

## Running Training

### Basic Training

```powershell
# Activate the virtual environment (if using uv)
.venv\Scripts\activate

# Run training
train --data-file data/input/livia_glucose.csv --max-steps 1000
```

### With Specific CPU Allocation

If you want to limit CPU usage (recommended on Windows):

```powershell
# Set before running
$env:RAY_NUM_CPUS = "4"
train --data-file data/input/livia_glucose.csv
```

## Common Windows Issues and Solutions

### Issue 1: Long Path Errors

**Problem**: Ray fails with "path too long" errors.

**Solution**: 
- Set `RAY_TMPDIR` to a short path (e.g., `C:\ray_tmp`)
- Enable long paths in Windows:
  ```powershell
  # Run as Administrator
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
  ```

### Issue 2: Ray Dashboard Not Starting

**Problem**: Ray dashboard fails to start or causes errors.

**Solution**: The `.rayconfig` disables the dashboard by default on Windows. This is intentional and won't affect training.

### Issue 3: Memory Issues

**Problem**: Training crashes with out-of-memory errors.

**Solution**:
- Reduce `max_steps` in training
- Reduce object store memory in `.rayconfig`
- Train fewer models at once
- Close other applications

### Issue 4: Plasma Store Errors

**Problem**: Errors related to Plasma object store.

**Solution**: The configuration uses a conservative 2GB object store. Adjust in `.rayconfig` based on your system:

```ini
[ray]
object_store_memory = 1000000000  # 1GB for systems with 8GB RAM
```

### Issue 5: Slow Performance

**Problem**: Training is slower on Windows than expected.

**Solution**:
- Consider using WSL2 (Windows Subsystem for Linux) for better performance
- Limit number of CPUs to avoid overhead
- Ensure antivirus is not scanning the ray_tmp directory

## Using WSL2 (Recommended for Advanced Users)

For better Ray performance, consider using WSL2:

### 1. Install WSL2

```powershell
# In PowerShell as Administrator
wsl --install
```

### 2. Install Python in WSL2

```bash
# In WSL2 terminal
sudo apt update
sudo apt install python3.13 python3-pip
pip3 install uv
```

### 3. Run Project in WSL2

```bash
cd /mnt/c/path/to/glucose-neuralforecast
uv sync
train --data-file data/input/livia_glucose.csv
```

Performance in WSL2 is typically much closer to native Linux performance.

## Verification

To verify your Ray installation:

```python
import ray
import platform

print(f"Platform: {platform.system()}")
print(f"Ray version: {ray.__version__}")

# Try initializing Ray
ray.init(include_dashboard=False)
print("Ray initialized successfully!")
print(f"Available resources: {ray.available_resources()}")
ray.shutdown()
```

Save as `test_ray.py` and run:

```powershell
python test_ray.py
```

## Resource Monitoring

On Windows, monitor Ray and training resources using:

```powershell
# CPU and Memory
Get-Process python

# Detailed system info
Get-ComputerInfo | Select-Object CsTotalPhysicalMemory, CsNumberOfProcessors
```

## Getting Help

If you encounter issues:

1. Check the [Ray Windows documentation](https://docs.ray.io/en/latest/ray-overview/installation.html)
2. Review error logs in the `logs/` directory
3. Try reducing resources (fewer CPUs, less memory)
4. Consider using WSL2 for better compatibility

## Performance Tips

1. **Close Unnecessary Applications**: Free up RAM before training
2. **Use SSD**: Store data and models on SSD for faster I/O
3. **Limit Models**: Train fewer models at once with `--models "NHITS,LSTM"`
4. **Reduce Steps**: Use fewer training steps for testing: `--max-steps 100`
5. **Monitor Resources**: Use Task Manager to watch CPU/RAM usage
6. **Disable Dashboard**: Keep `include_dashboard = false` in config

## References

- [Ray Installation Documentation](https://docs.ray.io/en/latest/ray-overview/installation.html)
- [Ray Windows Support](https://docs.ray.io/en/latest/ray-overview/installation.html#windows-support)
- [NeuralForecast Documentation](https://nixtla.github.io/neuralforecast/)

