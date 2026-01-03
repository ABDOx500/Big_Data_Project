"""
Prediction Module for Spam Detection

This module provides functions to:
- Load trained models from disk
- Predict whether a single message is spam or ham
- Predict on multiple messages (batch prediction)
- Get confidence scores for predictions

Usage:
    from spark_ml.prediction import SpamPredictor
    
    predictor = SpamPredictor(model_name="naive_bayes")
    result = predictor.predict_message("Free money! Click now!")
    print(result)  # {'prediction': 'spam', 'confidence': 0.95, ...}

Author: Spam Detection Team
Date: January 2025
"""

import os
import sys
import logging
import platform
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

# Configure Hadoop for Windows (required for Parquet file operations)
if platform.system() == "Windows":
    if not os.environ.get("HADOOP_HOME"):
        hadoop_home = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "hadoop")
        if os.path.exists(os.path.join(hadoop_home, "bin", "winutils.exe")):
            os.environ["HADOOP_HOME"] = hadoop_home
            hadoop_bin = os.path.join(hadoop_home, "bin")
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.functions import col, udf, lit
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.classification import (
    NaiveBayesModel,
    LogisticRegressionModel,
    RandomForestClassificationModel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    Data class to store prediction results.
    
    Attributes:
        message: The original message text
        prediction: Predicted label ("spam" or "ham")
        confidence: Confidence score (0.0 to 1.0)
        model_used: Name of the model that made the prediction
        label: Numeric label (1 for spam, 0 for ham)
    """
    message: str
    prediction: str
    confidence: float
    model_used: str
    label: int
    
    def to_dict(self) -> dict:
        """Convert prediction result to dictionary."""
        return {
            "message": self.message,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "model_used": self.model_used,
            "label": self.label
        }


class SpamPredictor:
    """
    A class to make spam/ham predictions using trained models.
    
    This class provides methods to load trained models and make
    predictions on single messages or batches of messages.
    
    Attributes:
        spark (SparkSession): The Spark session instance
        model_name (str): Name of the currently loaded model
        model: The loaded Spark ML model
        feature_pipeline (PipelineModel): Feature extraction pipeline
    """
    
    # Valid model names
    VALID_MODELS = ["naive_bayes", "logistic_regression", "random_forest"]
    
    def __init__(
        self,
        model_name: str = "naive_bayes",
        models_path: str = "models",
        app_name: str = "SpamDetection-Prediction"
    ):
        """
        Initialize the SpamPredictor.
        
        Args:
            model_name: Name of the model to load (default: "naive_bayes")
                       Options: "naive_bayes", "logistic_regression", "random_forest"
            models_path: Base path where models are stored
            app_name: Name for the Spark application
        """
        self.models_path = models_path
        self.model_name = model_name
        self.model = None
        self.feature_pipeline = None
        self._spark_initialized = False
        
        # Validate model name
        if model_name not in self.VALID_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. "
                f"Valid options: {self.VALID_MODELS}"
            )
        
        # Initialize Spark session
        logger.info("Initializing Spark session...")
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "4") \
            .getOrCreate()
        
        # Set log level to reduce verbosity
        self.spark.sparkContext.setLogLevel("WARN")
        self._spark_initialized = True
        logger.info("Spark session initialized successfully")
        
        # Load feature pipeline
        self._load_feature_pipeline()
        
        # Load the specified model
        self.load_model(model_name)
    
    def _load_feature_pipeline(self):
        """Load the feature extraction pipeline."""
        pipeline_path = os.path.join(self.models_path, "feature_pipeline")
        
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(
                f"Feature pipeline not found at {pipeline_path}. "
                "Please train models first using train_models.py"
            )
        
        logger.info(f"Loading feature pipeline from: {pipeline_path}")
        self.feature_pipeline = PipelineModel.load(pipeline_path)
        logger.info("Feature pipeline loaded successfully")
    
    def load_model(self, model_name: str):
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the model to load
                       Options: "naive_bayes", "logistic_regression", "random_forest"
        """
        if model_name not in self.VALID_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. "
                f"Valid options: {self.VALID_MODELS}"
            )
        
        model_path = os.path.join(self.models_path, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train models first using train_models.py"
            )
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load the appropriate model type
        if model_name == "naive_bayes":
            self.model = NaiveBayesModel.load(model_path)
        elif model_name == "logistic_regression":
            self.model = LogisticRegressionModel.load(model_path)
        elif model_name == "random_forest":
            self.model = RandomForestClassificationModel.load(model_path)
        
        self.model_name = model_name
        logger.info(f"Model '{model_name}' loaded successfully")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available trained models.
        
        Returns:
            List[str]: Names of available models
        """
        available = []
        for model_name in self.VALID_MODELS:
            model_path = os.path.join(self.models_path, model_name)
            if os.path.exists(model_path):
                available.append(model_name)
        return available
    
    def get_model_metrics(self) -> Optional[dict]:
        """
        Get metrics for all trained models.
        
        Returns:
            dict: Model metrics or None if not available
        """
        metrics_path = os.path.join(self.models_path, "metrics.json")
        
        if not os.path.exists(metrics_path):
            return None
        
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def _prepare_features(self, messages: List[str]) -> DataFrame:
        """
        Prepare features from raw messages.
        
        Args:
            messages: List of message texts
            
        Returns:
            DataFrame: Data with extracted features
        """
        from pyspark.sql.functions import length, size, split, lower, trim, regexp_replace
        
        # Create DataFrame from messages
        rows = [Row(text=msg) for msg in messages]
        df = self.spark.createDataFrame(rows)
        
        # Add basic preprocessing (similar to feature_engineering.py)
        df = df \
            .withColumn("text_clean", lower(col("text"))) \
            .withColumn("text_clean", regexp_replace(col("text_clean"), r"http\S+|www\S+", " ")) \
            .withColumn("text_clean", regexp_replace(col("text_clean"), r"[^\w\s]", " ")) \
            .withColumn("text_clean", regexp_replace(col("text_clean"), r"\s+", " ")) \
            .withColumn("text_clean", trim(col("text_clean")))
        
        # Add basic numeric features
        df = df \
            .withColumn("message_length", length(col("text"))) \
            .withColumn("word_count", size(split(col("text"), " ")))
        
        # Add custom features using inline functions (to avoid import issues)
        def count_special_chars(text):
            if text is None:
                return 0
            special_chars = set("!@#$%^&*()_+-=[]{}|;':\",.<>?~`\\")
            return sum(1 for char in text if char in special_chars)
        
        def count_digits(text):
            if text is None:
                return 0
            return sum(1 for char in text if char.isdigit())
        
        # Register UDFs with correct return types
        from pyspark.sql.types import IntegerType
        count_special_udf = udf(count_special_chars, IntegerType())
        count_digits_udf = udf(count_digits, IntegerType())
        
        df = df \
            .withColumn("special_char_count", count_special_udf(col("text"))) \
            .withColumn("digit_count", count_digits_udf(col("text")))
        
        # Apply feature pipeline (TF-IDF transformation)
        df_features = self.feature_pipeline.transform(df)
        
        return df_features
    
    def predict_message(
        self,
        message: str,
        model_name: Optional[str] = None
    ) -> PredictionResult:
        """
        Predict whether a single message is spam or ham.
        
        Args:
            message: The message text to classify
            model_name: Optional model to use (uses current model if None)
            
        Returns:
            PredictionResult: Prediction with confidence score
        """
        # Switch model if specified
        if model_name and model_name != self.model_name:
            self.load_model(model_name)
        
        # Prepare features
        df = self._prepare_features([message])
        
        # Make prediction
        predictions = self.model.transform(df)
        
        # Extract results
        result_row = predictions.select(
            "text",
            "prediction",
            "probability"
        ).collect()[0]
        
        # Get prediction and confidence
        prediction_label = int(result_row["prediction"])
        prediction_text = "spam" if prediction_label == 1 else "ham"
        
        # Get probability (confidence)
        probability = result_row["probability"]
        # For binary classification, probability is [P(ham), P(spam)]
        # We want confidence in the predicted class
        confidence = float(probability[prediction_label])
        
        result = PredictionResult(
            message=message,
            prediction=prediction_text,
            confidence=confidence,
            model_used=self.model_name,
            label=prediction_label
        )
        
        logger.info(f"Prediction: {prediction_text} (confidence: {confidence:.4f})")
        
        return result
    
    def predict_batch(
        self,
        messages: List[str],
        model_name: Optional[str] = None
    ) -> List[PredictionResult]:
        """
        Predict on multiple messages at once.
        
        More efficient than calling predict_message() multiple times
        as it processes all messages in a single Spark job.
        
        Args:
            messages: List of message texts to classify
            model_name: Optional model to use (uses current model if None)
            
        Returns:
            List[PredictionResult]: List of prediction results
        """
        if not messages:
            return []
        
        # Switch model if specified
        if model_name and model_name != self.model_name:
            self.load_model(model_name)
        
        logger.info(f"Batch prediction on {len(messages)} messages...")
        
        # Prepare features
        df = self._prepare_features(messages)
        
        # Make predictions
        predictions = self.model.transform(df)
        
        # Collect results
        results = []
        for row in predictions.select("text", "prediction", "probability").collect():
            prediction_label = int(row["prediction"])
            prediction_text = "spam" if prediction_label == 1 else "ham"
            probability = row["probability"]
            confidence = float(probability[prediction_label])
            
            results.append(PredictionResult(
                message=row["text"],
                prediction=prediction_text,
                confidence=confidence,
                model_used=self.model_name,
                label=prediction_label
            ))
        
        # Summary stats
        spam_count = sum(1 for r in results if r.prediction == "spam")
        ham_count = len(results) - spam_count
        logger.info(f"Batch complete: {spam_count} spam, {ham_count} ham")
        
        return results
    
    def predict_dataframe(
        self,
        df: DataFrame,
        text_column: str = "text",
        model_name: Optional[str] = None
    ) -> DataFrame:
        """
        Predict on a Spark DataFrame.
        
        Useful for processing large datasets that don't fit in memory.
        
        Args:
            df: Input DataFrame with text column
            text_column: Name of the column containing message text
            model_name: Optional model to use (uses current model if None)
            
        Returns:
            DataFrame: Input DataFrame with prediction columns added
        """
        # Switch model if specified
        if model_name and model_name != self.model_name:
            self.load_model(model_name)
        
        logger.info(f"DataFrame prediction using column: {text_column}")
        
        # Rename column if needed
        if text_column != "text":
            df = df.withColumn("text", col(text_column))
        
        # Add preprocessing
        from pyspark.sql.functions import length, size, split, lower, trim, regexp_replace
        from pyspark.sql.types import IntegerType
        
        df = df \
            .withColumn("text_clean", lower(col("text"))) \
            .withColumn("text_clean", regexp_replace(col("text_clean"), r"http\S+|www\S+", " ")) \
            .withColumn("text_clean", regexp_replace(col("text_clean"), r"[^\w\s]", " ")) \
            .withColumn("text_clean", regexp_replace(col("text_clean"), r"\s+", " ")) \
            .withColumn("text_clean", trim(col("text_clean")))
        
        # Add numeric features
        df = df \
            .withColumn("message_length", length(col("text"))) \
            .withColumn("word_count", size(split(col("text"), " ")))
        
        # Add custom features
        def count_special_chars(text):
            if text is None:
                return 0
            special_chars = set("!@#$%^&*()_+-=[]{}|;':\",./<>?~`\\")
            return sum(1 for char in text if char in special_chars)
        
        def count_digits(text):
            if text is None:
                return 0
            return sum(1 for char in text if char.isdigit())
        
        count_special_udf = udf(count_special_chars, IntegerType())
        count_digits_udf = udf(count_digits, IntegerType())
        
        df = df \
            .withColumn("special_char_count", count_special_udf(col("text"))) \
            .withColumn("digit_count", count_digits_udf(col("text")))
        
        # Apply feature pipeline
        df_features = self.feature_pipeline.transform(df)
        
        # Make predictions
        df_predictions = self.model.transform(df_features)
        
        # Add human-readable prediction label
        from pyspark.sql.functions import when
        df_predictions = df_predictions.withColumn(
            "prediction_label",
            when(col("prediction") == 1, "spam").otherwise("ham")
        )
        
        logger.info("DataFrame prediction complete")
        
        return df_predictions
    
    def get_current_model_info(self) -> dict:
        """
        Get information about the currently loaded model.
        
        Returns:
            dict: Model information
        """
        metrics = self.get_model_metrics()
        model_metrics = metrics.get(self.model_name, {}) if metrics else {}
        
        return {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "accuracy": model_metrics.get("accuracy"),
            "f1_score": model_metrics.get("f1_score"),
            "is_best_model": metrics.get("best_model") == self.model_name if metrics else None
        }
    
    def stop(self):
        """Stop the Spark session."""
        if self._spark_initialized and self.spark:
            self.spark.stop()
            self._spark_initialized = False
            logger.info("Spark session stopped")


def predict_message(
    message: str,
    model_name: str = "naive_bayes",
    models_path: str = "models"
) -> dict:
    """
    Convenience function to predict a single message.
    
    Creates a predictor, makes prediction, and cleans up.
    For multiple predictions, use SpamPredictor class directly.
    
    Args:
        message: Message text to classify
        model_name: Model to use
        models_path: Path to saved models
        
    Returns:
        dict: Prediction result
    """
    predictor = SpamPredictor(model_name=model_name, models_path=models_path)
    try:
        result = predictor.predict_message(message)
        return result.to_dict()
    finally:
        predictor.stop()


def predict_batch(
    messages: List[str],
    model_name: str = "naive_bayes",
    models_path: str = "models"
) -> List[dict]:
    """
    Convenience function to predict multiple messages.
    
    Creates a predictor, makes predictions, and cleans up.
    
    Args:
        messages: List of message texts
        model_name: Model to use
        models_path: Path to saved models
        
    Returns:
        List[dict]: List of prediction results
    """
    predictor = SpamPredictor(model_name=model_name, models_path=models_path)
    try:
        results = predictor.predict_batch(messages)
        return [r.to_dict() for r in results]
    finally:
        predictor.stop()


def main():
    """
    Main function to demonstrate prediction functionality.
    
    Can be executed directly:
        python -m spark_ml.prediction
        python -m spark_ml.prediction "Your message here"
        python -m spark_ml.prediction "Your message" --model logistic_regression
    """
    import argparse
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Spam Detection Predictor")
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="Message to classify"
    )
    parser.add_argument(
        "--model", "-m",
        default="naive_bayes",
        choices=["naive_bayes", "logistic_regression", "random_forest"],
        help="Model to use for prediction"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with sample messages"
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SpamPredictor(model_name=args.model, models_path="models")
    
    try:
        # Show model info
        model_info = predictor.get_current_model_info()
        print("\n" + "="*60)
        print("SPAM DETECTION PREDICTOR")
        print("="*60)
        print(f"Model: {model_info['model_name']}")
        print(f"Type: {model_info['model_type']}")
        if model_info['f1_score']:
            print(f"F1-Score: {model_info['f1_score']:.4f}")
        print("="*60)
        
        if args.message:
            # Single message prediction
            result = predictor.predict_message(args.message)
            print(f"\nMessage: {result.message[:100]}...")
            print(f"Prediction: {result.prediction.upper()}")
            print(f"Confidence: {result.confidence:.4f} ({result.confidence*100:.1f}%)")
            
        elif args.demo:
            # Demo with sample messages
            demo_messages = [
                "Hey, are we still meeting for lunch tomorrow?",
                "CONGRATULATIONS! You've won a FREE iPhone! Click here NOW!",
                "Can you pick up some milk on your way home?",
                "URGENT: Your account has been compromised. Click to verify.",
                "Thanks for the birthday wishes!",
                "Win £1000 cash! Text WIN to 12345 to claim your prize!",
                "Meeting rescheduled to 3pm. See you then.",
                "FREE entry to win a brand new car! Reply YES now!",
            ]
            
            print("\nDEMO: Classifying sample messages\n")
            print("-"*60)
            
            results = predictor.predict_batch(demo_messages)
            
            for result in results:
                status = "[SPAM]" if result.prediction == "spam" else "[HAM] "
                conf = f"{result.confidence*100:.1f}%"
                msg = result.message[:50] + "..." if len(result.message) > 50 else result.message
                print(f"{status} ({conf:>6}): {msg}")
            
            print("-"*60)
            spam_count = sum(1 for r in results if r.prediction == "spam")
            print(f"\nTotal: {spam_count} spam, {len(results) - spam_count} ham")
            
        else:
            # Interactive mode
            print("\nEnter messages to classify (Ctrl+C to exit):\n")
            
            while True:
                try:
                    message = input("Message: ").strip()
                    if message:
                        result = predictor.predict_message(message)
                        status = "[SPAM]" if result.prediction == "spam" else "[HAM] "
                        print(f"  -> {status} (confidence: {result.confidence:.4f})\n")
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break
        
        print("\n")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        predictor.stop()


if __name__ == "__main__":
    main()
