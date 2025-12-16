"""
Data Preprocessing Module for Spam Detection

This module handles:
- Loading raw CSV data into Spark DataFrame
- Cleaning data (remove duplicates, handle missing values)
- Converting labels (spam -> 1, ham -> 0)
- Train/test splitting
- Saving processed data as Parquet

Author: Spam Detection Team
Date: January 2025
"""

import os
import sys
import logging
import platform
from typing import Tuple, Optional

# Configure Hadoop for Windows (required for Parquet file operations)
# This sets HADOOP_HOME to use winutils.exe from a temp directory
if platform.system() == "Windows":
    # Check if HADOOP_HOME is already set
    if not os.environ.get("HADOOP_HOME"):
        # Use the hadoop binaries in temp folder
        hadoop_home = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "hadoop")
        if os.path.exists(os.path.join(hadoop_home, "bin", "winutils.exe")):
            os.environ["HADOOP_HOME"] = hadoop_home
            # Add hadoop bin to PATH
            hadoop_bin = os.path.join(hadoop_home, "bin")
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, 
    when, 
    length, 
    trim, 
    lower,
    regexp_replace,
    count as spark_count
)
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class to handle data preprocessing for spam detection.
    
    This class provides methods to load, clean, transform, and save
    SMS spam data using PySpark.
    
    Attributes:
        spark (SparkSession): The Spark session instance
        raw_data_path (str): Path to the raw CSV file
        processed_data_path (str): Path to save processed Parquet files
    """
    
    def __init__(
        self, 
        raw_data_path: str = "data/raw/spam.csv",
        processed_data_path: str = "data/processed",
        app_name: str = "SpamDetection-Preprocessing"
    ):
        """
        Initialize the DataPreprocessor.
        
        Args:
            raw_data_path: Path to the raw spam.csv file
            processed_data_path: Directory to save processed data
            app_name: Name for the Spark application
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        
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
    
    def load_raw_data(self) -> DataFrame:
        """
        Load the raw CSV data into a Spark DataFrame.
        
        The SMS Spam Collection dataset has the format:
        label,message (with possible extra empty columns)
        
        Returns:
            DataFrame: Raw data with 'label' and 'message' columns
            
        Raises:
            FileNotFoundError: If the raw data file doesn't exist
        """
        logger.info(f"Loading raw data from: {self.raw_data_path}")
        
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_data_path}")
        
        # Define schema for the SMS Spam Collection dataset
        # The dataset has: label, message, and some empty columns
        schema = StructType([
            StructField("label", StringType(), True),
            StructField("message", StringType(), True),
            StructField("_c2", StringType(), True),  # Empty column
            StructField("_c3", StringType(), True),  # Empty column
            StructField("_c4", StringType(), True)   # Empty column
        ])
        
        # Load CSV with proper handling of quoted fields
        df = self.spark.read.csv(
            self.raw_data_path,
            header=True,
            schema=schema,
            quote='"',
            escape='"'
        )
        
        # Select only relevant columns and rename
        df = df.select(
            col("label").alias("original_label"),
            col("message").alias("text")
        )
        
        row_count = df.count()
        logger.info(f"Loaded {row_count} rows from raw data")
        
        return df
    
    def clean_data(self, df: DataFrame) -> DataFrame:
        """
        Clean the data by removing duplicates and handling missing values.
        
        Cleaning steps:
        1. Remove rows with null/empty labels or messages
        2. Trim whitespace from text
        3. Remove duplicate messages
        4. Filter out very short messages (< 3 characters)
        
        Args:
            df: Raw DataFrame with 'original_label' and 'text' columns
            
        Returns:
            DataFrame: Cleaned data
        """
        logger.info("Starting data cleaning...")
        
        initial_count = df.count()
        logger.info(f"Initial row count: {initial_count}")
        
        # Step 1: Remove null values
        df_clean = df.filter(
            col("original_label").isNotNull() & 
            col("text").isNotNull()
        )
        after_null = df_clean.count()
        logger.info(f"After removing nulls: {after_null} (removed {initial_count - after_null})")
        
        # Step 2: Trim whitespace and filter empty messages
        df_clean = df_clean.withColumn("text", trim(col("text")))
        df_clean = df_clean.filter(length(col("text")) >= 3)
        after_empty = df_clean.count()
        logger.info(f"After removing empty/short messages: {after_empty}")
        
        # Step 3: Normalize labels to lowercase
        df_clean = df_clean.withColumn(
            "original_label", 
            lower(trim(col("original_label")))
        )
        
        # Step 4: Filter only valid labels (spam or ham)
        df_clean = df_clean.filter(
            col("original_label").isin(["spam", "ham"])
        )
        after_valid = df_clean.count()
        logger.info(f"After filtering valid labels: {after_valid}")
        
        # Step 5: Remove duplicates based on message text
        df_clean = df_clean.dropDuplicates(["text"])
        final_count = df_clean.count()
        logger.info(f"After removing duplicates: {final_count}")
        
        logger.info(f"Data cleaning complete. Removed {initial_count - final_count} rows total.")
        
        return df_clean
    
    def convert_labels(self, df: DataFrame) -> DataFrame:
        """
        Convert string labels to numeric values.
        
        Conversion:
        - 'spam' -> 1
        - 'ham' -> 0
        
        Args:
            df: DataFrame with 'original_label' column
            
        Returns:
            DataFrame: Data with numeric 'label' column added
        """
        logger.info("Converting labels to numeric format...")
        
        df_labeled = df.withColumn(
            "label",
            when(col("original_label") == "spam", 1).otherwise(0).cast(IntegerType())
        )
        
        # Verify conversion
        label_counts = df_labeled.groupBy("label", "original_label").count()
        logger.info("Label conversion complete:")
        label_counts.show()
        
        return df_labeled
    
    def show_balance_stats(self, df: DataFrame) -> dict:
        """
        Display and return class balance statistics.
        
        Args:
            df: DataFrame with 'label' column
            
        Returns:
            dict: Statistics including counts and ratios
        """
        logger.info("\n" + "="*50)
        logger.info("CLASS BALANCE STATISTICS")
        logger.info("="*50)
        
        total = df.count()
        spam_count = df.filter(col("label") == 1).count()
        ham_count = df.filter(col("label") == 0).count()
        
        spam_ratio = (spam_count / total) * 100
        ham_ratio = (ham_count / total) * 100
        imbalance_ratio = ham_count / spam_count if spam_count > 0 else float('inf')
        
        stats = {
            "total_messages": total,
            "spam_count": spam_count,
            "ham_count": ham_count,
            "spam_percentage": spam_ratio,
            "ham_percentage": ham_ratio,
            "imbalance_ratio": imbalance_ratio
        }
        
        logger.info(f"Total Messages: {total}")
        logger.info(f"Spam Messages:  {spam_count} ({spam_ratio:.2f}%)")
        logger.info(f"Ham Messages:   {ham_count} ({ham_ratio:.2f}%)")
        logger.info(f"Imbalance Ratio (Ham:Spam): {imbalance_ratio:.2f}:1")
        logger.info("="*50 + "\n")
        
        return stats
    
    def add_basic_features(self, df: DataFrame) -> DataFrame:
        """
        Add basic text features that might be useful for classification.
        
        Features added:
        - message_length: Character count of the message
        - word_count: Number of words in the message
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            DataFrame: Data with additional feature columns
        """
        logger.info("Adding basic text features...")
        
        from pyspark.sql.functions import size, split
        
        df_features = df \
            .withColumn("message_length", length(col("text"))) \
            .withColumn("word_count", size(split(col("text"), " ")))
        
        # Show feature statistics
        logger.info("Feature statistics:")
        df_features.select("message_length", "word_count").describe().show()
        
        return df_features
    
    def split_data(
        self, 
        df: DataFrame, 
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Split data into training and test sets.
        
        Args:
            df: DataFrame to split
            train_ratio: Proportion of data for training (default 0.8)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple[DataFrame, DataFrame]: (train_df, test_df)
        """
        logger.info(f"Splitting data: {train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% test")
        
        train_df, test_df = df.randomSplit(
            [train_ratio, 1 - train_ratio], 
            seed=seed
        )
        
        train_count = train_df.count()
        test_count = test_df.count()
        
        logger.info(f"Training set: {train_count} samples")
        logger.info(f"Test set: {test_count} samples")
        
        # Check balance in both sets
        train_spam = train_df.filter(col("label") == 1).count()
        test_spam = test_df.filter(col("label") == 1).count()
        
        logger.info(f"Training spam ratio: {train_spam/train_count*100:.2f}%")
        logger.info(f"Test spam ratio: {test_spam/test_count*100:.2f}%")
        
        return train_df, test_df
    
    def save_processed_data(
        self, 
        df: DataFrame, 
        filename: str = "cleaned_data"
    ) -> str:
        """
        Save processed data as Parquet files.
        
        Args:
            df: DataFrame to save
            filename: Name for the output file (without extension)
            
        Returns:
            str: Path where data was saved
        """
        output_path = os.path.join(self.processed_data_path, filename)
        
        logger.info(f"Saving processed data to: {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # Save as Parquet with overwrite mode
        df.write.mode("overwrite").parquet(output_path)
        
        logger.info(f"Data saved successfully to {output_path}")
        
        return output_path
    
    def save_as_csv(
        self, 
        df: DataFrame, 
        filename: str = "cleaned_data.csv"
    ) -> str:
        """
        Save processed data as a single CSV file.
        
        This is useful for inspection and use outside of Spark.
        
        Args:
            df: DataFrame to save
            filename: Name for the output CSV file
            
        Returns:
            str: Path where CSV was saved
        """
        output_path = os.path.join(self.processed_data_path, filename)
        
        logger.info(f"Saving data as CSV to: {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # Coalesce to single partition for single CSV file, then save
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(
            output_path + "_temp"
        )
        
        # Rename the part file to the desired filename
        import glob
        temp_dir = output_path + "_temp"
        part_files = glob.glob(os.path.join(temp_dir, "part-*.csv"))
        
        if part_files:
            import shutil
            # Copy the part file to final destination
            shutil.copy(part_files[0], output_path)
            # Remove temp directory
            shutil.rmtree(temp_dir)
            logger.info(f"CSV saved successfully to {output_path}")
        else:
            logger.warning("Could not find part file to rename")
        
        return output_path
    
    def save_train_test_split(
        self, 
        train_df: DataFrame, 
        test_df: DataFrame
    ) -> Tuple[str, str]:
        """
        Save train and test datasets separately.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple[str, str]: (train_path, test_path)
        """
        train_path = self.save_processed_data(train_df, "train_data")
        test_path = self.save_processed_data(test_df, "test_data")
        
        return train_path, test_path
    
    def process_all(self) -> Tuple[DataFrame, DataFrame, dict]:
        """
        Execute the complete preprocessing pipeline.
        
        Pipeline steps:
        1. Load raw data
        2. Clean data
        3. Convert labels
        4. Add basic features
        5. Show statistics
        6. Split into train/test
        7. Save processed data
        
        Returns:
            Tuple[DataFrame, DataFrame, dict]: (train_df, test_df, stats)
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING DATA PREPROCESSING PIPELINE")
        logger.info("="*60 + "\n")
        
        # Step 1: Load raw data
        raw_df = self.load_raw_data()
        
        # Step 2: Clean data
        clean_df = self.clean_data(raw_df)
        
        # Step 3: Convert labels
        labeled_df = self.convert_labels(clean_df)
        
        # Step 4: Add basic features
        featured_df = self.add_basic_features(labeled_df)
        
        # Step 5: Show statistics
        stats = self.show_balance_stats(featured_df)
        
        # Step 6: Split data
        train_df, test_df = self.split_data(featured_df)
        
        # Step 7: Save processed data (Parquet format for Spark)
        self.save_processed_data(featured_df, "cleaned_data")
        self.save_train_test_split(train_df, test_df)
        
        # Step 8: Also save as CSV for easy inspection
        self.save_as_csv(featured_df, "cleaned_data.csv")
        
        logger.info("\n" + "="*60)
        logger.info("DATA PREPROCESSING COMPLETE")
        logger.info("="*60 + "\n")
        
        # Show sample of processed data
        logger.info("Sample of processed data:")
        featured_df.select("text", "label", "message_length", "word_count").show(5, truncate=50)
        
        return train_df, test_df, stats
    
    def stop(self):
        """Stop the Spark session."""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")


def main():
    """
    Main function to run the preprocessing pipeline.
    
    Can be executed directly:
        python -m spark_ml.data_preprocessing
    """
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Change to project root for relative paths
    os.chdir(project_root)
    
    logger.info(f"Project root: {project_root}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        raw_data_path="data/raw/spam.csv",
        processed_data_path="data/processed"
    )
    
    try:
        # Run full preprocessing pipeline
        train_df, test_df, stats = preprocessor.process_all()
        
        # Print summary
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Total messages processed: {stats['total_messages']}")
        print(f"Spam messages: {stats['spam_count']} ({stats['spam_percentage']:.1f}%)")
        print(f"Ham messages: {stats['ham_count']} ({stats['ham_percentage']:.1f}%)")
        print(f"Class imbalance ratio: {stats['imbalance_ratio']:.2f}:1 (Ham:Spam)")
        print("="*60)
        print("\nProcessed data saved to:")
        print("  - data/processed/cleaned_data/       (Parquet)")
        print("  - data/processed/train_data/         (Parquet)")
        print("  - data/processed/test_data/          (Parquet)")
        print("  - data/processed/cleaned_data.csv    (CSV)")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise
        
    finally:
        preprocessor.stop()


if __name__ == "__main__":
    main()
