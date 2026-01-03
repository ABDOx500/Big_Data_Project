"""
Model Training Module for Spam Detection

This module handles training of three machine learning models using Spark MLlib:
1. Naive Bayes - Probabilistic classifier based on Bayes' theorem
2. Logistic Regression - Linear model with sigmoid activation
3. Random Forest - Ensemble of decision trees

Each model is:
- Trained on preprocessed data with TF-IDF features
- Evaluated using accuracy, precision, recall, and F1-score
- Saved to the models/ directory for later use
- Compared to select the best performing model

Author: Spam Detection Team
Date: January 2025
"""

import os
import sys
import logging
import platform
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import json

# Configure Hadoop for Windows (required for Parquet file operations)
if platform.system() == "Windows":
    if not os.environ.get("HADOOP_HOME"):
        hadoop_home = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "hadoop")
        if os.path.exists(os.path.join(hadoop_home, "bin", "winutils.exe")):
            os.environ["HADOOP_HOME"] = hadoop_home
            hadoop_bin = os.path.join(hadoop_home, "bin")
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    NaiveBayes,
    LogisticRegression,
    RandomForestClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """
    Data class to store model evaluation metrics.
    
    Attributes:
        accuracy: Overall accuracy (correct predictions / total)
        precision: True positives / (True positives + False positives)
        recall: True positives / (True positives + False negatives)
        f1_score: Harmonic mean of precision and recall
        auc_roc: Area under ROC curve
    """
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc
        }


class SpamModelTrainer:
    """
    A class to train and evaluate spam detection models.
    
    This class provides methods to train three different classifiers,
    evaluate their performance, and select the best model.
    
    Models trained:
    1. Naive Bayes - Fast, probabilistic, good baseline
    2. Logistic Regression - Interpretable, balanced performance
    3. Random Forest - Robust, handles non-linear patterns
    
    Attributes:
        spark (SparkSession): The Spark session instance
        models (dict): Dictionary of trained models
        metrics (dict): Dictionary of model metrics
        best_model_name (str): Name of the best performing model
    """
    
    def __init__(self, app_name: str = "SpamDetection-ModelTraining"):
        """
        Initialize the SpamModelTrainer.
        
        Args:
            app_name: Name for the Spark application
        """
        self.models = {}
        self.metrics = {}
        self.best_model_name = None
        
        # Initialize Spark session
        logger.info("Initializing Spark session...")
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "4") \
            .getOrCreate()
        
        # Set log level to reduce verbosity
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark session initialized successfully")
    
    def _evaluate_model(
        self,
        model: PipelineModel,
        test_data: DataFrame,
        model_name: str
    ) -> ModelMetrics:
        """
        Evaluate a trained model on test data.
        
        Computes multiple metrics:
        - Accuracy: Overall correctness
        - Precision: Spam detection exactness
        - Recall: Spam detection completeness
        - F1-Score: Balance of precision and recall
        - AUC-ROC: Overall model discrimination ability
        
        Args:
            model: Trained Spark ML model
            test_data: Test DataFrame with features and label
            model_name: Name of the model for logging
            
        Returns:
            ModelMetrics: Object containing all evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        predictions = model.transform(test_data)
        
        # Binary classification evaluator for AUC-ROC
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        # Multiclass evaluator for other metrics
        accuracy_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        
        precision_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )
        
        recall_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedRecall"
        )
        
        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )
        
        # Compute metrics
        metrics = ModelMetrics(
            model_name=model_name,
            accuracy=accuracy_evaluator.evaluate(predictions),
            precision=precision_evaluator.evaluate(predictions),
            recall=recall_evaluator.evaluate(predictions),
            f1_score=f1_evaluator.evaluate(predictions),
            auc_roc=binary_evaluator.evaluate(predictions)
        )
        
        logger.info(f"{model_name} Metrics:")
        logger.info(f"  Accuracy:  {metrics.accuracy:.4f}")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall:    {metrics.recall:.4f}")
        logger.info(f"  F1-Score:  {metrics.f1_score:.4f}")
        logger.info(f"  AUC-ROC:   {metrics.auc_roc:.4f}")
        
        return metrics
    
    def train_naive_bayes(
        self,
        train_data: DataFrame,
        test_data: DataFrame,
        smoothing: float = 1.0
    ) -> Tuple[PipelineModel, ModelMetrics]:
        logger.info("\n" + "="*60)
        logger.info("TRAINING NAIVE BAYES CLASSIFIER")
        logger.info("="*60)
        
        # Create Naive Bayes classifier
        nb = NaiveBayes(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            smoothing=smoothing,
            modelType="multinomial"  # For TF-IDF features
        )
        
        # Train the model
        logger.info("Training Naive Bayes model...")
        model = nb.fit(train_data)
        
        # Evaluate
        metrics = self._evaluate_model(model, test_data, "Naive Bayes")
        
        # Store model and metrics
        self.models["naive_bayes"] = model
        self.metrics["naive_bayes"] = metrics
        
        return model, metrics
    
    def train_logistic_regression(
        self,
        train_data: DataFrame,
        test_data: DataFrame,
        max_iter: int = 100,
        reg_param: float = 0.01
    ) -> Tuple[PipelineModel, ModelMetrics]:
        logger.info("\n" + "="*60)
        logger.info("TRAINING LOGISTIC REGRESSION CLASSIFIER")
        logger.info("="*60)
        
        # Create Logistic Regression classifier
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            maxIter=max_iter,
            regParam=reg_param,
            elasticNetParam=0.0  # L2 only
        )
        
        # Train the model
        logger.info("Training Logistic Regression model...")
        model = lr.fit(train_data)
        
        # Log model summary
        logger.info(f"Number of iterations: {model.summary.totalIterations}")
        
        # Evaluate
        metrics = self._evaluate_model(model, test_data, "Logistic Regression")
        
        # Store model and metrics
        self.models["logistic_regression"] = model
        self.metrics["logistic_regression"] = metrics
        
        return model, metrics
    
    def train_random_forest(
        self,
        train_data: DataFrame,
        test_data: DataFrame,
        num_trees: int = 100,
        max_depth: int = 10
    ) -> Tuple[PipelineModel, ModelMetrics]:
        logger.info("\n" + "="*60)
        logger.info("TRAINING RANDOM FOREST CLASSIFIER")
        logger.info("="*60)
        
        # Create Random Forest classifier
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            numTrees=num_trees,
            maxDepth=max_depth,
            seed=42
        )
        
        # Train the model
        logger.info(f"Training Random Forest model with {num_trees} trees...")
        model = rf.fit(train_data)
        
        # Log feature importance summary
        importances = model.featureImportances
        logger.info(f"Feature vector size: {importances.size}")
        
        # Evaluate
        metrics = self._evaluate_model(model, test_data, "Random Forest")
        
        # Store model and metrics
        self.models["random_forest"] = model
        self.metrics["random_forest"] = metrics
        
        return model, metrics
    
    def train_with_cross_validation(
        self,
        train_data: DataFrame,
        model_name: str = "logistic_regression",
        num_folds: int = 3
    ) -> Tuple[PipelineModel, float]:
        """
        Train a model with cross-validation for hyperparameter tuning.
        
        Uses k-fold cross-validation to find the best hyperparameters
        and returns the best model.
        
        Args:
            train_data: Training DataFrame
            model_name: Which model to tune ("naive_bayes", "logistic_regression", "random_forest")
            num_folds: Number of cross-validation folds
            
        Returns:
            Tuple[PipelineModel, float]: Best model and cross-validation score
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"CROSS-VALIDATION FOR {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        # Create model and parameter grid based on model type
        if model_name == "naive_bayes":
            model = NaiveBayes(
                featuresCol="features",
                labelCol="label",
                modelType="multinomial"
            )
            param_grid = ParamGridBuilder() \
                .addGrid(model.smoothing, [0.5, 1.0, 2.0]) \
                .build()
                
        elif model_name == "logistic_regression":
            model = LogisticRegression(
                featuresCol="features",
                labelCol="label"
            )
            param_grid = ParamGridBuilder() \
                .addGrid(model.regParam, [0.001, 0.01, 0.1]) \
                .addGrid(model.maxIter, [50, 100]) \
                .build()
                
        elif model_name == "random_forest":
            model = RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                seed=42
            )
            param_grid = ParamGridBuilder() \
                .addGrid(model.numTrees, [50, 100]) \
                .addGrid(model.maxDepth, [5, 10]) \
                .build()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create evaluator (using F1-score)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )
        
        # Create cross-validator
        cv = CrossValidator(
            estimator=model,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=num_folds,
            parallelism=2
        )
        
        # Run cross-validation
        logger.info(f"Running {num_folds}-fold cross-validation...")
        logger.info(f"Testing {len(param_grid)} parameter combinations...")
        
        cv_model = cv.fit(train_data)
        
        # Get best model and score
        best_model = cv_model.bestModel
        avg_metrics = cv_model.avgMetrics
        best_score = max(avg_metrics)
        
        logger.info(f"Best cross-validation F1-score: {best_score:.4f}")
        
        return best_model, best_score
    
    def select_best_model(self) -> str:
        """
        Select the best model based on F1-score.
        
        F1-score is used because it balances precision and recall,
        which is important for spam detection where both false positives
        (legitimate mail marked as spam) and false negatives (spam not caught)
        are costly.
        
        Returns:
            str: Name of the best model
        """
        logger.info("\n" + "="*60)
        logger.info("SELECTING BEST MODEL")
        logger.info("="*60)
        
        if not self.metrics:
            raise ValueError("No models trained yet. Train models first.")
        
        # Find model with highest F1-score
        best_model = max(
            self.metrics.items(),
            key=lambda x: x[1].f1_score
        )
        
        self.best_model_name = best_model[0]
        
        logger.info(f"Best model: {self.best_model_name}")
        logger.info(f"F1-Score: {best_model[1].f1_score:.4f}")
        
        return self.best_model_name
    
    def print_comparison_table(self):
        """
        Print a comparison table of all trained models.
        
        Displays accuracy, precision, recall, F1-score, and AUC-ROC
        for each model in a formatted table.
        """
        if not self.metrics:
            logger.warning("No models trained yet.")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC-ROC':<12}")
        print("-"*80)
        
        for name, metrics in self.metrics.items():
            is_best = " *" if name == self.best_model_name else ""
            print(f"{name + is_best:<25} {metrics.accuracy:<12.4f} {metrics.precision:<12.4f} "
                  f"{metrics.recall:<12.4f} {metrics.f1_score:<12.4f} {metrics.auc_roc:<12.4f}")
        
        print("-"*80)
        print("* Best model (highest F1-score)")
        print("="*80 + "\n")
    
    def save_model(self, model_name: str, path: str) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            path: Directory path to save the model
            
        Returns:
            str: Path where model was saved
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        logger.info(f"Saving {model_name} to: {path}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Save model
        model.write().overwrite().save(path)
        
        logger.info(f"Model saved successfully to {path}")
        return path
    
    def save_all_models(self, base_path: str = "models") -> Dict[str, str]:
        """
        Save all trained models to disk.
        
        Args:
            base_path: Base directory for saving models
            
        Returns:
            Dict[str, str]: Dictionary mapping model names to save paths
        """
        paths = {}
        
        for model_name in self.models.keys():
            path = os.path.join(base_path, model_name)
            self.save_model(model_name, path)
            paths[model_name] = path
        
        return paths
    
    def save_metrics(self, path: str = "models/metrics.json") -> str:
        """
        Save model metrics to a JSON file.
        
        Args:
            path: Path to save the metrics file
            
        Returns:
            str: Path where metrics were saved
        """
        metrics_dict = {
            name: metrics.to_dict() 
            for name, metrics in self.metrics.items()
        }
        metrics_dict["best_model"] = self.best_model_name
        
        # Create directory if needed
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Metrics saved to {path}")
        return path
    
    def train_all(
        self,
        train_data: DataFrame,
        test_data: DataFrame
    ) -> str:
        """
        Train all three models and select the best one.
        
        This is the main method that:
        1. Trains Naive Bayes
        2. Trains Logistic Regression
        3. Trains Random Forest
        4. Evaluates all models
        5. Selects the best model
        6. Prints comparison table
        
        Args:
            train_data: Training DataFrame with features and label
            test_data: Test DataFrame for evaluation
            
        Returns:
            str: Name of the best model
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*60)
        
        # Train all models
        self.train_naive_bayes(train_data, test_data)
        self.train_logistic_regression(train_data, test_data)
        self.train_random_forest(train_data, test_data)
        
        # Select best model
        best_model = self.select_best_model()
        
        # Print comparison
        self.print_comparison_table()
        
        return best_model
    
    def stop(self):
        """Stop the Spark session."""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


def main():
    """
    Main function to train all spam detection models.
    
    Can be executed directly:
        python -m spark_ml.train_models
    """
    # Import feature engineering module
    from spark_ml.feature_engineering import FeatureEngineer
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    logger.info(f"Project root: {project_root}")
    
    # Initialize trainer
    trainer = SpamModelTrainer()
    
    try:
        # Load processed data
        train_data_path = "data/processed/train_data"
        test_data_path = "data/processed/test_data"
        
        if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
            logger.error("Processed data not found. Please run data_preprocessing.py first.")
            return
        
        logger.info("Loading training data...")
        train_df = trainer.spark.read.parquet(train_data_path)
        
        logger.info("Loading test data...")
        test_df = trainer.spark.read.parquet(test_data_path)
        
        logger.info(f"Training samples: {train_df.count()}")
        logger.info(f"Test samples: {test_df.count()}")
        
        # Initialize feature engineer to transform data
        engineer = FeatureEngineer(num_features=10000)
        engineer.spark = trainer.spark  # Reuse spark session
        
        # Check if pipeline exists, otherwise build new one
        pipeline_path = "models/feature_pipeline"
        if os.path.exists(pipeline_path):
            logger.info("Loading existing feature pipeline...")
            engineer.load_pipeline(pipeline_path)
        else:
            logger.info("Building new feature pipeline...")
            engineer.build_full_pipeline()
            engineer.fit_pipeline(train_df)
        
        # Transform data
        logger.info("Transforming training data...")
        train_features = engineer.fit_transform(train_df)
        
        logger.info("Transforming test data...")
        test_features = engineer.transform(test_df)
        
        # Cache data for faster training
        train_features.cache()
        test_features.cache()
        
        # Train all models
        best_model = trainer.train_all(train_features, test_features)
        
        # Save all models
        logger.info("\nSaving models...")
        trainer.save_all_models("models")
        
        # Save metrics
        trainer.save_metrics("models/metrics.json")
        
        # Print final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best model: {best_model}")
        print(f"Best F1-Score: {trainer.metrics[best_model].f1_score:.4f}")
        print("\nModels saved to:")
        print("  - models/naive_bayes/")
        print("  - models/logistic_regression/")
        print("  - models/random_forest/")
        print("  - models/metrics.json")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        trainer.stop()


if __name__ == "__main__":
    main()
