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

# Lazy imports to avoid circular dependencies
# Import modules explicitly when needed:
#   from spark_ml.data_preprocessing import DataPreprocessor
#   from spark_ml.feature_engineering import FeatureEngineer
#   from spark_ml.train_models import SpamModelTrainer
#   from spark_ml.prediction import SpamPredictor

__all__ = [
    "data_preprocessing",
    "feature_engineering", 
    "train_models",
    "prediction"
]
