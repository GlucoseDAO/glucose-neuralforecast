"""
Ray initialization utilities for Windows-specific configuration.

This module provides utilities for configuring Ray environment
and initializing Ray with Windows-optimized settings.
"""

import os
import platform
from pathlib import Path
from typing import Optional
from eliot import start_action


def configure_ray_environment() -> None:
    """
    Configure environment variables for Ray on Windows.
    
    Sets up Windows-specific Ray environment variables if running on Windows
    and they're not already set.
    """
    with start_action(action_type="configure_ray_environment"):
        if platform.system() != "Windows":
            return
        
        # Set RAY_TMPDIR if not already set
        if "RAY_TMPDIR" not in os.environ:
            ray_tmp = Path.home() / "ray_tmp"
            os.environ["RAY_TMPDIR"] = str(ray_tmp)
            
        # Enable Windows cluster support
        if "RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER" not in os.environ:
            os.environ["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] = "1"
            
        # Disable log deduplication for better debugging
        if "RAY_DEDUP_LOGS" not in os.environ:
            os.environ["RAY_DEDUP_LOGS"] = "0"


def init_ray_for_training(
    num_cpus: Optional[int] = None,
    object_store_memory: Optional[int] = None,
    include_dashboard: bool = False,
    verbose: bool = False
) -> None:
    """
    Initialize Ray with Windows-optimized settings for training.
    
    Parameters
    ----------
    num_cpus : int, optional
        Number of CPUs to allocate to Ray. If None, uses all available.
    object_store_memory : int, optional
        Object store memory in bytes. If None, uses Ray's default (30% of system memory).
    include_dashboard : bool, default False
        Whether to include Ray dashboard. Disabled by default on Windows due to potential issues.
    verbose : bool, default False
        Whether to print Ray initialization details.
        
    Note
    ----
    This function is optional. NeuralForecast typically handles Ray initialization internally.
    Only use this if you need explicit control over Ray settings.
    """
    with start_action(action_type="init_ray_for_training", num_cpus=num_cpus, verbose=verbose):
        try:
            import ray
            
            # Check if Ray is already initialized
            if ray.is_initialized():
                if verbose:
                    print("Ray is already initialized")
                return
            
            # Configure environment first
            configure_ray_environment()
            
            # Build initialization kwargs
            init_kwargs = {
                "include_dashboard": include_dashboard,
                "ignore_reinit_error": True,
            }
            
            if num_cpus is not None:
                init_kwargs["num_cpus"] = num_cpus
                
            if object_store_memory is not None:
                init_kwargs["object_store_memory"] = object_store_memory
            
            # Initialize Ray
            ray.init(**init_kwargs)
            
            if verbose:
                print(f"Ray initialized on {platform.system()}")
                print(f"Ray version: {ray.__version__}")
                print(f"Available resources: {ray.available_resources()}")
                
        except ImportError:
            if verbose:
                print("Ray not installed, skipping Ray initialization")
        except Exception as e:
            if verbose:
                print(f"Failed to initialize Ray: {e}")


def shutdown_ray() -> None:
    """
    Shutdown Ray if it's running.
    
    Useful for cleanup after training.
    """
    with start_action(action_type="shutdown_ray"):
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except ImportError:
            pass


def get_ray_info() -> dict:
    """
    Get information about Ray initialization status and resources.
    
    Returns
    -------
    dict
        Dictionary containing Ray status information.
    """
    with start_action(action_type="get_ray_info") as action:
        info = {
            "platform": platform.system(),
            "ray_installed": False,
            "ray_initialized": False,
            "ray_version": None,
            "available_resources": None,
        }
        
        try:
            import ray
            info["ray_installed"] = True
            info["ray_version"] = ray.__version__
            
            if ray.is_initialized():
                info["ray_initialized"] = True
                info["available_resources"] = ray.available_resources()
        except ImportError:
            pass
        
        action.add_success_fields(**info)
        return info

