"""
Feature Engineering Module for Spam Detection

This module handles:
- Text preprocessing pipeline (Tokenization, Stop Words Removal)
- TF-IDF feature extraction (HashingTF + IDF)
- Additional feature engineering (message length, special chars)
- Building complete Spark ML Pipeline

Author: Spam Detection Team
Date: January 2025
"""

import os
import sys
import logging
import platform
from typing import Tuple, Optional, List

# Configure Hadoop for Windows (required for Parquet file operations)
if platform.system() == "Windows":
    if not os.environ.get("HADOOP_HOME"):
        hadoop_home = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "hadoop")
        if os.path.exists(os.path.join(hadoop_home, "bin", "winutils.exe")):
            os.environ["HADOOP_HOME"] = hadoop_home
            hadoop_bin = os.path.join(hadoop_home, "bin")
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, 
    length, 
    udf,
    regexp_replace,
    lower,
    trim
)
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    VectorAssembler,
    StringIndexer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Standalone helper functions for UDFs (must be outside class to serialize)
# ============================================================================

def count_special_chars(text: str) -> int:
    """
    Count special characters in text.
    
    Special characters include: !@#$%^&*()_+-=[]{}|;':",.<>?/~`
    
    Args:
        text: Input text string
        
    Returns:
        int: Count of special characters
    """
    if text is None:
        return 0
    special_chars = set("!@#$%^&*()_+-=[]{}|;':\",./<>?~`\\")
    return sum(1 for char in text if char in special_chars)


def count_uppercase_ratio(text: str) -> float:
    """
    Calculate the ratio of uppercase letters in text.
    
    Args:
        text: Input text string
        
    Returns:
        float: Ratio of uppercase letters (0.0 to 1.0)
    """
    if text is None or len(text) == 0:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0.0
    uppercase = sum(1 for c in letters if c.isupper())
    return uppercase / len(letters)


def count_digits(text: str) -> int:
    """
    Count digits in text.
    
    Args:
        text: Input text string
        
    Returns:
        int: Count of digits
    """
    if text is None:
        return 0
    return sum(1 for char in text if char.isdigit())


class FeatureEngineer:
    """
    A class to handle feature engineering for spam detection.
    
    This class provides methods to build and fit a complete text
    preprocessing and feature extraction pipeline using Spark ML.
    
    Attributes:
        spark (SparkSession): The Spark session instance
        num_features (int): Number of features for HashingTF
        pipeline (Pipeline): The Spark ML pipeline
        fitted_pipeline (PipelineModel): The fitted pipeline model
    """
    
    def __init__(
        self,
        num_features: int = 10000,
        app_name: str = "SpamDetection-FeatureEngineering"
    ):
        """
        Initialize the FeatureEngineer.
        
        Args:
            num_features: Number of features for HashingTF (default 10000)
            app_name: Name for the Spark application
        """
        self.num_features = num_features
        self.pipeline = None
        self.fitted_pipeline = None
        
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
    
    def add_custom_features(self, df: DataFrame) -> DataFrame:
        """
        Add custom engineered features to the DataFrame.
        
        Features added:
        - message_length: Character count (if not already present)
        - special_char_count: Number of special characters
        - uppercase_ratio: Ratio of uppercase letters
        - digit_count: Number of digits
        - word_count: Number of words (if not already present)
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            DataFrame: Data with additional feature columns
        """
        logger.info("Adding custom engineered features...")
        
        # Register UDFs using standalone functions (not class methods, for serialization)
        count_special_udf = udf(count_special_chars, IntegerType())
        uppercase_ratio_udf = udf(count_uppercase_ratio, DoubleType())
        count_digits_udf = udf(count_digits, IntegerType())
        
        # Add features
        df_features = df
        
        # Add message_length if not present
        if "message_length" not in df.columns:
            df_features = df_features.withColumn("message_length", length(col("text")))
        
        # Add word_count if not present
        if "word_count" not in df.columns:
            from pyspark.sql.functions import size, split
            df_features = df_features.withColumn("word_count", size(split(col("text"), " ")))
        
        # Add new custom features
        df_features = df_features \
            .withColumn("special_char_count", count_special_udf(col("text"))) \
            .withColumn("uppercase_ratio", uppercase_ratio_udf(col("text"))) \
            .withColumn("digit_count", count_digits_udf(col("text")))
        
        logger.info("Custom features added successfully")
        
        # Show feature statistics
        logger.info("Custom feature statistics:")
        df_features.select(
            "message_length", 
            "word_count", 
            "special_char_count", 
            "uppercase_ratio",
            "digit_count"
        ).describe().show()
        
        return df_features
    
    def build_text_pipeline(self) -> Pipeline:
        """
        Build the text preprocessing pipeline.
        
        Pipeline stages:
        1. Tokenizer: Split text into words
        2. StopWordsRemover: Remove common stop words
        3. HashingTF: Convert words to term frequency vectors
        4. IDF: Apply inverse document frequency weighting
        
        Returns:
            Pipeline: Unfitted Spark ML Pipeline
        """
        logger.info("Building text preprocessing pipeline...")
        
        # Stage 1: Tokenizer - split text into words
        tokenizer = Tokenizer(
            inputCol="text_clean",
            outputCol="words"
        )
        
        # Stage 2: Remove stop words
        stop_words_remover = StopWordsRemover(
            inputCol="words",
            outputCol="filtered_words"
        )
        
        # Stage 3: HashingTF - convert words to term frequency vectors
        hashing_tf = HashingTF(
            inputCol="filtered_words",
            outputCol="raw_features",
            numFeatures=self.num_features
        )
        
        # Stage 4: IDF - apply inverse document frequency
        idf = IDF(
            inputCol="raw_features",
            outputCol="tfidf_features",
            minDocFreq=5  # Ignore terms that appear in fewer than 5 documents
        )
        
        # Create pipeline
        self.pipeline = Pipeline(stages=[
            tokenizer,
            stop_words_remover,
            hashing_tf,
            idf
        ])
        
        logger.info(f"Pipeline built with {len(self.pipeline.getStages())} stages")
        return self.pipeline
    
    def build_full_pipeline(self, include_label_indexer: bool = False) -> Pipeline:
        """
        Build the complete feature engineering pipeline including
        custom features and vector assembly.
        
        Pipeline stages:
        1. Tokenizer
        2. StopWordsRemover
        3. HashingTF
        4. IDF
        5. VectorAssembler (combines TF-IDF with custom features)
        6. StringIndexer for label (optional)
        
        Args:
            include_label_indexer: Whether to include label indexing
            
        Returns:
            Pipeline: Complete unfitted Spark ML Pipeline
        """
        logger.info("Building complete feature engineering pipeline...")
        
        # Stage 1: Tokenizer
        tokenizer = Tokenizer(
            inputCol="text_clean",
            outputCol="words"
        )
        
        # Stage 2: Remove stop words
        stop_words_remover = StopWordsRemover(
            inputCol="words",
            outputCol="filtered_words"
        )
        
        # Stage 3: HashingTF
        hashing_tf = HashingTF(
            inputCol="filtered_words",
            outputCol="raw_features",
            numFeatures=self.num_features
        )
        
        # Stage 4: IDF
        idf = IDF(
            inputCol="raw_features",
            outputCol="tfidf_features",
            minDocFreq=5
        )
        
        # Stage 5: Vector Assembler - combine all features
        # Note: Custom features should already be added to DataFrame
        assembler = VectorAssembler(
            inputCols=[
                "tfidf_features",
                "message_length",
                "word_count",
                "special_char_count",
                "digit_count"
            ],
            outputCol="features"
        )
        
        stages = [tokenizer, stop_words_remover, hashing_tf, idf, assembler]
        
        # Optional: Add label indexer
        if include_label_indexer:
            label_indexer = StringIndexer(
                inputCol="original_label",
                outputCol="label_indexed"
            )
            stages.append(label_indexer)
        
        self.pipeline = Pipeline(stages=stages)
        
        logger.info(f"Complete pipeline built with {len(stages)} stages")
        return self.pipeline
    
    def preprocess_text(self, df: DataFrame) -> DataFrame:
        """
        Preprocess text column for the pipeline.
        
        Preprocessing steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove extra whitespace
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            DataFrame: Data with cleaned 'text_clean' column
        """
        logger.info("Preprocessing text...")
        
        df_clean = df \
            .withColumn("text_clean", lower(col("text"))) \
            .withColumn("text_clean", regexp_replace(col("text_clean"), r"http\S+|www\S+", " ")) \
            .withColumn("text_clean", regexp_replace(col("text_clean"), r"[^\w\s]", " ")) \
            .withColumn("text_clean", regexp_replace(col("text_clean"), r"\s+", " ")) \
            .withColumn("text_clean", trim(col("text_clean")))
        
        logger.info("Text preprocessing complete")
        return df_clean
    
    def fit_pipeline(self, df: DataFrame) -> PipelineModel:
        """
        Fit the pipeline on the training data.
        
        Args:
            df: Training DataFrame (must have 'text' column)
            
        Returns:
            PipelineModel: Fitted pipeline model
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_text_pipeline() or build_full_pipeline() first.")
        
        logger.info("Fitting pipeline on training data...")
        
        # Preprocess text
        df_preprocessed = self.preprocess_text(df)
        
        # Add custom features
        df_features = self.add_custom_features(df_preprocessed)
        
        # Fit the pipeline
        self.fitted_pipeline = self.pipeline.fit(df_features)
        
        logger.info("Pipeline fitted successfully")
        return self.fitted_pipeline
    
    def transform(self, df: DataFrame) -> DataFrame:
        """
        Transform data using the fitted pipeline.
        
        Args:
            df: DataFrame to transform (must have 'text' column)
            
        Returns:
            DataFrame: Transformed data with features
        """
        if self.fitted_pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_pipeline() first.")
        
        logger.info("Transforming data...")
        
        # Preprocess text
        df_preprocessed = self.preprocess_text(df)
        
        # Add custom features
        df_features = self.add_custom_features(df_preprocessed)
        
        # Transform using fitted pipeline
        df_transformed = self.fitted_pipeline.transform(df_features)
        
        logger.info("Data transformation complete")
        return df_transformed
    
    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Fit the pipeline and transform the data in one step.
        
        Args:
            df: Training DataFrame (must have 'text' column)
            
        Returns:
            DataFrame: Transformed data with features
        """
        if self.pipeline is None:
            self.build_full_pipeline()
        
        logger.info("Fitting and transforming data...")
        
        # Preprocess text
        df_preprocessed = self.preprocess_text(df)
        
        # Add custom features
        df_features = self.add_custom_features(df_preprocessed)
        
        # Fit and transform
        self.fitted_pipeline = self.pipeline.fit(df_features)
        df_transformed = self.fitted_pipeline.transform(df_features)
        
        logger.info("Fit-transform complete")
        return df_transformed
    
    def save_pipeline(self, path: str) -> str:
        """
        Save the fitted pipeline to disk.
        
        Args:
            path: Directory path to save the pipeline
            
        Returns:
            str: Path where pipeline was saved
        """
        if self.fitted_pipeline is None:
            raise ValueError("No fitted pipeline to save. Call fit_pipeline() first.")
        
        logger.info(f"Saving pipeline to: {path}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Save pipeline
        self.fitted_pipeline.write().overwrite().save(path)
        
        logger.info(f"Pipeline saved successfully to {path}")
        return path
    
    def load_pipeline(self, path: str) -> PipelineModel:
        """
        Load a fitted pipeline from disk.
        
        Args:
            path: Directory path where pipeline is saved
            
        Returns:
            PipelineModel: Loaded pipeline model
        """
        logger.info(f"Loading pipeline from: {path}")
        
        self.fitted_pipeline = PipelineModel.load(path)
        
        logger.info("Pipeline loaded successfully")
        return self.fitted_pipeline
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features.
        
        Returns:
            List[str]: List of feature names
        """
        base_features = [
            f"tfidf_{i}" for i in range(self.num_features)
        ]
        custom_features = [
            "message_length",
            "word_count", 
            "special_char_count",
            "digit_count"
        ]
        return base_features + custom_features
    
    def stop(self):
        """Stop the Spark session."""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


def create_feature_pipeline(
    train_df: DataFrame,
    num_features: int = 10000
) -> Tuple[PipelineModel, DataFrame]:
    """
    Convenience function to create and fit a feature pipeline.
    
    Args:
        train_df: Training DataFrame with 'text' and 'label' columns
        num_features: Number of TF-IDF features (default 10000)
        
    Returns:
        Tuple[PipelineModel, DataFrame]: (fitted_pipeline, transformed_data)
    """
    engineer = FeatureEngineer(num_features=num_features)
    engineer.build_full_pipeline()
    transformed_df = engineer.fit_transform(train_df)
    return engineer.fitted_pipeline, transformed_df


def main():
    """
    Main function to demonstrate feature engineering.
    
    Can be executed directly:
        python -m spark_ml.feature_engineering
    """
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    logger.info(f"Project root: {project_root}")
    
    # Initialize feature engineer
    engineer = FeatureEngineer(num_features=10000)
    
    try:
        # Load processed training data
        train_data_path = "data/processed/train_data"
        
        if not os.path.exists(train_data_path):
            logger.error(f"Training data not found at {train_data_path}")
            logger.error("Please run data_preprocessing.py first")
            return
        
        logger.info(f"Loading training data from: {train_data_path}")
        train_df = engineer.spark.read.parquet(train_data_path)
        
        logger.info(f"Loaded {train_df.count()} training samples")
        
        # Build and fit the pipeline
        engineer.build_full_pipeline()
        transformed_df = engineer.fit_transform(train_df)
        
        # Show sample of transformed data
        logger.info("\nSample of transformed data:")
        transformed_df.select(
            "text",
            "label",
            "message_length",
            "special_char_count",
            "features"
        ).show(5, truncate=50)
        
        # Save the fitted pipeline
        pipeline_path = "models/feature_pipeline"
        engineer.save_pipeline(pipeline_path)
        
        # Print summary
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Training samples processed: {train_df.count()}")
        print(f"Number of TF-IDF features: {engineer.num_features}")
        print(f"Custom features added: 5")
        print(f"  - message_length")
        print(f"  - word_count")
        print(f"  - special_char_count")
        print(f"  - uppercase_ratio")
        print(f"  - digit_count")
        print(f"Total feature vector size: {engineer.num_features + 4}")
        print("="*60)
        print(f"\nFeature pipeline saved to: {pipeline_path}/")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        raise
        
    finally:
        engineer.stop()


if __name__ == "__main__":
    main()
