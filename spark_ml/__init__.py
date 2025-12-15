"""
Spam Detection Spark ML Package

This package contains modules for:
- Data preprocessing
- Feature engineering
- Model training
- Prediction
"""

__version__ = "1.0.0"
__author__ = "Spam Detection Team"

from . import data_preprocessing
from . import feature_engineering
from . import train_models
from . import prediction

__all__ = [
    "data_preprocessing",
    "feature_engineering", 
    "train_models",
    "prediction"
]
