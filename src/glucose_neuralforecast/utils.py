"""Utility functions for the glucose_neuralforecast package."""

from pathlib import Path


def resolve_base_folder() -> Path:
    """
    Resolve the base folder of the project.
    If running from notebooks, go up one level, otherwise use current directory.
    
    Returns:
        Path: The base folder of the project
    """
    current = Path.cwd()
    if current.name == 'notebooks':
        return current.parent
    # Also check if we're in src or a subdirectory
    if 'src' in current.parts:
        # Go up to the project root
        while current.name != 'glucose-neuralforecast' and current.parent != current:
            current = current.parent
        return current
    return current

