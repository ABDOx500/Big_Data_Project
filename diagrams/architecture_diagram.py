"""
Spam Detection System - Architecture Diagram

This script generates a colorful, vertical architecture diagram 
of the spam detection project using the Python 'diagrams' library.

Requirements:
    pip install diagrams

Usage:
    python architecture_diagram.py

Output:
    spam_detection_architecture.png

Author: Big Data Project
Date: January 2026
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.programming.framework import Flask
from diagrams.onprem.analytics import Spark
from diagrams.onprem.client import Users
from diagrams.generic.storage import Storage
from diagrams.generic.database import SQL
import os

# Change to the script's directory to save the output there
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Graph attributes for better visualization - HORIZONTAL layout with bold fonts
graph_attr = {
    "fontsize": "24",
    "fontname": "Helvetica Bold",
    "bgcolor": "#f5f5f5",
    "pad": "0.8",
    "splines": "spline",
    "nodesep": "1.0",
    "ranksep": "1.2",
    "dpi": "150"
}

node_attr = {
    "fontsize": "11",
    "fontname": "Helvetica Bold",
}

edge_attr = {
    "fontsize": "10",
    "fontname": "Helvetica Bold",
    "color": "#555555"
}

with Diagram(
    "Spam Detection System Architecture",
    filename="spam_detection_architecture",
    show=False,
    direction="LR",  # Left to Right (HORIZONTAL)
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr
):
    # User Layer
    with Cluster("User Interface", graph_attr={"bgcolor": "#e3f2fd", "fontcolor": "#1565c0", "style": "rounded", "fontname": "Helvetica Bold"}):
        user = Users("End Users")
    
    # Web Application Layer
    with Cluster("Web Application Layer", graph_attr={"bgcolor": "#e8f5e9", "fontcolor": "#2e7d32", "style": "rounded", "fontname": "Helvetica Bold"}):
        with Cluster("Frontend", graph_attr={"bgcolor": "#c8e6c9", "fontcolor": "#1b5e20"}):
            frontend = Python("HTML / CSS / JS\nInteractive UI")
        
        with Cluster("Backend API", graph_attr={"bgcolor": "#a5d6a7", "fontcolor": "#1b5e20"}):
            flask_app = Flask("Flask REST API\n(app.py)")
    
    # Machine Learning Layer - Spark with ML Models inside
    with Cluster("Apache Spark ML Pipeline", graph_attr={"bgcolor": "#fff3e0", "fontcolor": "#e65100", "style": "rounded", "fontname": "Helvetica Bold"}):
        with Cluster("Spark Core Components", graph_attr={"bgcolor": "#ffe0b2", "fontcolor": "#bf360c"}):
            feature_eng = Spark("TF-IDF\nText Vectorization")
            predictor = Spark("SpamPredictor\nReal-time Classification")
        
        with Cluster("ML Models", graph_attr={"bgcolor": "#ffcc80", "fontcolor": "#bf360c"}):
            naive_bayes = Python("Naive Bayes\nAccuracy: 98.49%")
            log_reg = Python("Logistic Regression\nAccuracy: 97.59%")
            random_forest = Python("Random Forest\nAccuracy: 89.96%")
    
    # Data Layer
    with Cluster("Data Storage Layer", graph_attr={"bgcolor": "#fce4ec", "fontcolor": "#c2185b", "style": "rounded", "fontname": "Helvetica Bold"}):
        with Cluster("Raw Data", graph_attr={"bgcolor": "#f8bbd9", "fontcolor": "#880e4f"}):
            raw_data = Storage("SMS Spam Dataset\n(5,574 messages)")
        
        with Cluster("Processed Data", graph_attr={"bgcolor": "#f48fb1", "fontcolor": "#880e4f"}):
            processed = SQL("Preprocessed Data\n(Parquet Format)")
        
        with Cluster("Trained Models", graph_attr={"bgcolor": "#ec407a", "fontcolor": "#ffffff"}):
            models = Storage("Serialized Models\n(/models)")

    # Define the flow with colored edges
    # User to Frontend
    user >> Edge(label="HTTP Request", color="#1565c0", style="bold") >> frontend
    
    # Frontend to Flask
    frontend >> Edge(label="API Call", color="#2e7d32", style="bold") >> flask_app
    
    # Flask to Prediction Engine
    flask_app >> Edge(label="Predict", color="#e65100", style="bold") >> predictor
    
    # Prediction flow
    predictor >> Edge(label="Extract Features", color="#ff6f00") >> feature_eng
    feature_eng >> Edge(label="Classify", color="#ff6f00") >> naive_bayes
    
    # Model connections (best model highlighted)
    naive_bayes - Edge(style="dotted", color="#757575") - log_reg
    log_reg - Edge(style="dotted", color="#757575") - random_forest
    
    # Data pipeline
    raw_data >> Edge(label="Preprocess", color="#c2185b") >> processed
    processed >> Edge(label="Train", color="#c2185b") >> models
    models >> Edge(label="Load Model", color="#c2185b", style="dashed") >> predictor
    
    # Response flow (dashed lines going back up)
    predictor >> Edge(label="spam/ham", color="#4caf50", style="dashed") >> flask_app
    flask_app >> Edge(label="JSON Response", color="#4caf50", style="dashed") >> frontend
    frontend >> Edge(label="Display Result", color="#4caf50", style="dashed") >> user

print("[SUCCESS] Architecture diagram generated: spam_detection_architecture.png")
