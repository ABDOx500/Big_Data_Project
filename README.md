# 📧 Spam Detection System with PySpark ML

A complete machine learning pipeline for real-time spam detection in SMS/email messages, built using Apache Spark MLlib.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PySpark](https://img.shields.io/badge/PySpark-3.5.0-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Performance Metrics](#-performance-metrics)
- [Future Improvements](#-future-improvements)
- [License](#-license)

## 🎯 Project Overview

This project implements a spam detection system using Apache Spark's MLlib for scalable machine learning. The system can classify SMS/email messages as spam or ham (not spam) in real-time using multiple trained models.

### Objectives

1. **Data Processing**: Preprocess and clean the SMS Spam Collection dataset
2. **Feature Engineering**: Extract meaningful features from text data using NLP techniques
3. **Model Training**: Train and compare multiple ML models (Naive Bayes, Logistic Regression, Random Forest)
4. **Deployment**: Provide a web interface and REST API for real-time predictions

## ✨ Features

- **Multiple ML Models**: Compare Naive Bayes, Logistic Regression, and Random Forest classifiers
- **Scalable Processing**: Built on Apache Spark for distributed data processing
- **Real-time Predictions**: Web interface and REST API for instant spam classification
- **Batch Processing**: Classify multiple messages at once
- **Model Comparison**: Visualize and compare model performance metrics
- **Interactive Notebooks**: Explore data and experiment with models

## 📁 Project Structure

```
spam-detection-spark/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── data/
│   ├── raw/                     # Original dataset
│   │   └── spam.csv
│   └── processed/               # Preprocessed data
│       └── cleaned_data.parquet
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Data analysis and visualization
│   └── 02_model_experiments.ipynb   # Model training experiments
├── spark_ml/
│   ├── __init__.py              # Package initialization
│   ├── data_preprocessing.py    # Data cleaning and preparation
│   ├── feature_engineering.py   # Feature extraction pipeline
│   ├── train_models.py          # Model training and evaluation
│   └── prediction.py            # Prediction interface
├── models/
│   ├── naive_bayes/             # Saved Naive Bayes model
│   ├── logistic_regression/     # Saved Logistic Regression model
│   └── random_forest/           # Saved Random Forest model
├── web_app/
│   ├── app.py                   # Flask application
│   ├── static/
│   │   ├── css/style.css        # Stylesheet
│   │   └── js/main.js           # Frontend JavaScript
│   └── templates/
│       └── index.html           # Web interface
└── scripts/
    ├── download_data.py         # Dataset download script
    └── train.py                 # Complete training pipeline
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Java 8 or 11 (required for Spark)
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/spam-detection-spark.git
cd spam-detection-spark
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Java (if not installed)

Spark requires Java. Download and install from [Oracle JDK](https://www.oracle.com/java/technologies/downloads/) or use OpenJDK.

Set the `JAVA_HOME` environment variable:

```bash
# Windows (PowerShell)
$env:JAVA_HOME = "C:\Program Files\Java\jdk-11"

# Linux/Mac
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
```

## 🚀 Quick Start

### Option 1: Full Training Pipeline

Run the complete pipeline (download data, preprocess, train models):

```bash
python scripts/train.py
```

### Option 2: Step-by-Step

```bash
# 1. Download dataset
python scripts/download_data.py

# 2. Preprocess data
python -m spark_ml.data_preprocessing

# 3. Train models
python -m spark_ml.train_models
```

### Option 3: Start Web Application

```bash
cd web_app
python app.py
```

Visit `http://localhost:5000` in your browser.

## 📊 Dataset

### SMS Spam Collection Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Size**: 5,574 messages
- **Distribution**: ~87% ham, ~13% spam
- **Format**: Tab-separated (label, message)

### Sample Data

| Label | Message |
|-------|---------|
| ham   | Go until jurong point, crazy.. Available only in bugis n great world la e buffet... |
| spam  | Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 |

## 🧠 Model Architecture

### Text Processing Pipeline

```
Raw Text → Tokenization → Stop Words Removal → TF-IDF → Feature Vector
```

1. **Tokenization**: Split text into individual words
2. **Stop Words Removal**: Remove common words (the, is, at, etc.)
3. **HashingTF**: Convert words to term frequency vectors
4. **IDF**: Apply inverse document frequency weighting

### Additional Features

- Message length (character count)
- Special character count
- Numeric character ratio

### Machine Learning Models

| Model | Description |
|-------|-------------|
| **Naive Bayes** | Probabilistic classifier based on Bayes' theorem |
| **Logistic Regression** | Linear model for binary classification |
| **Random Forest** | Ensemble of decision trees |

## 💻 Usage

### Python API

```python
from spark_ml.prediction import SpamPredictor

# Initialize predictor
predictor = SpamPredictor(model_name="logistic_regression")

# Single message prediction
result = predictor.predict_message("Congratulations! You won a free iPhone!")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
messages = [
    "Hey, are we still meeting for lunch?",
    "URGENT: Your account has been compromised. Click here now!"
]
results = predictor.predict_batch(messages)
```

### Command Line

```bash
# Single prediction
python -m spark_ml.prediction "Your message here"

# Using specific model
python -m spark_ml.prediction "Your message" --model naive_bayes
```

## 📡 API Documentation

### Base URL

`http://localhost:5000`

### Endpoints

#### `GET /`

Returns the web interface.

#### `POST /predict`

Classify a single message.

**Request Body:**
```json
{
    "message": "Free money! Click now!",
    "model": "logistic_regression"
}
```

**Response:**
```json
{
    "prediction": "spam",
    "confidence": 0.95,
    "model_used": "logistic_regression",
    "processing_time_ms": 12
}
```

#### `POST /batch-predict`

Classify multiple messages.

**Request Body:**
```json
{
    "messages": ["Hello there!", "Win free prizes now!"],
    "model": "logistic_regression"
}
```

**Response:**
```json
{
    "predictions": [
        {"message": "Hello there!", "prediction": "ham", "confidence": 0.98},
        {"message": "Win free prizes now!", "prediction": "spam", "confidence": 0.92}
    ],
    "total_spam": 1,
    "total_ham": 1
}
```

#### `GET /stats`

Get model statistics.

**Response:**
```json
{
    "total_predictions": 1234,
    "spam_count": 156,
    "ham_count": 1078,
    "available_models": ["naive_bayes", "logistic_regression", "random_forest"],
    "current_model": "logistic_regression"
}
```

## 📈 Performance Metrics

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 97.2% | 95.8% | 91.3% | 93.5% |
| Logistic Regression | 97.8% | 96.4% | 93.2% | 94.7% |
| Random Forest | 97.5% | 97.1% | 90.8% | 93.8% |

*Note: Results may vary slightly based on random seed and data split.*

### Best Model Selection

**Logistic Regression** is selected as the default model based on the highest F1-score, providing the best balance between precision and recall.

## 🔮 Future Improvements

1. **Deep Learning**: Implement LSTM/Transformer models for better accuracy
2. **Multi-language Support**: Extend to support multiple languages
3. **Email Integration**: Direct email inbox monitoring
4. **User Feedback Loop**: Allow users to correct predictions for model improvement
5. **Model Versioning**: Implement MLflow for model tracking
6. **Containerization**: Docker support for easier deployment
7. **Cloud Deployment**: AWS/GCP deployment scripts

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Academic Project** | Big Data & Machine Learning | January 2025

For questions or support, please open an issue on GitHub.
