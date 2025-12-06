"""
Utility functions for AbMelt pipeline.
"""

import numpy as np
from typing import Optional, Union


# Default statistics for each temperature type
# Training set stats
DEFAULT_STATS = {
    "tagg": {"mean": 75.4, "std": 12.5},
    "tm": {"mean": 64.9, "std": 9.02},
    "tmon": {"mean": 56.0, "std": 9.55},
}


def normalize(
    arr: np.ndarray,
    temp_type: str,
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> np.ndarray:
    """
    Normalize a numpy array using z-score normalization.
    
    Formula: (x - mean) / std
    
    Args:
        arr: Input numpy array to normalize
        temp_type: Temperature type, one of "tagg", "tm", or "tmon"
        mean: Optional mean value. If None, uses default for temp_type
        std: Optional standard deviation value. If None, uses default for temp_type
    
    Returns:
        Normalized numpy array (z-scores)
    
    Raises:
        ValueError: If temp_type is not recognized
    """
    if temp_type not in DEFAULT_STATS:
        raise ValueError(
            f"Unknown temp_type: {temp_type}. Must be one of {list(DEFAULT_STATS.keys())}"
        )
    
    # Use defaults if not provided
    if mean is None:
        mean = DEFAULT_STATS[temp_type]["mean"]
    if std is None:
        std = DEFAULT_STATS[temp_type]["std"]
    
    if std == 0:
        raise ValueError("Standard deviation cannot be zero")
    
    return (arr - mean) / std


def renormalize(
    arr: np.ndarray,
    temp_type: str,
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> np.ndarray:
    """
    Renormalize a numpy array from z-scores back to original scale.
    
    Formula: x * std + mean
    
    Args:
        arr: Input numpy array of z-scores to renormalize
        temp_type: Temperature type, one of "tagg", "tm", or "tmon"
        mean: Optional mean value. If None, uses default for temp_type
        std: Optional standard deviation value. If None, uses default for temp_type
    
    Returns:
        Renormalized numpy array in original scale
    
    Raises:
        ValueError: If temp_type is not recognized
    """
    if temp_type not in DEFAULT_STATS:
        raise ValueError(
            f"Unknown temp_type: {temp_type}. Must be one of {list(DEFAULT_STATS.keys())}"
        )
    
    # Use defaults if not provided
    if mean is None:
        mean = DEFAULT_STATS[temp_type]["mean"]
    if std is None:
        std = DEFAULT_STATS[temp_type]["std"]
    
    return arr * std + mean

