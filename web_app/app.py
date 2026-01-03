"""
Flask Web Application for Spam Detection

This module provides a web interface and REST API for spam detection using
trained Spark ML models. It includes endpoints for single message prediction,
batch prediction, and statistics.

Routes:
    GET  /           - Main web interface
    POST /predict    - Classify a single message
    POST /batch-predict - Classify multiple messages
    GET  /stats      - Get model statistics
    GET  /models     - Get available models
    POST /switch-model - Switch to a different model

Author: Big Data Project
Date: January 2025
"""

import os
import sys
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add parent directory to path to import spark_ml modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from spark_ml.prediction import SpamPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Global variables
predictor: Optional[SpamPredictor] = None
prediction_history: List[Dict] = []
stats = {
    'total_predictions': 0,
    'spam_count': 0,
    'ham_count': 0,
    'start_time': datetime.now().isoformat()
}


def initialize_predictor():
    """
    Initialize the SpamPredictor with the best model.
    
    Loads the trained model from the models directory.
    Falls back to available models if the default is not found.
    """
    global predictor
    
    try:
        # Get the models directory path
        models_path = Path(__file__).parent.parent / 'models'
        
        # Try to load the best model (naive_bayes has highest accuracy: 98.49%)
        logger.info("Initializing SpamPredictor...")
        predictor = SpamPredictor(
            model_name='naive_bayes',
            models_path=str(models_path)
        )
        logger.info(f"Successfully loaded model: {predictor.model_name}")
        
    except Exception as e:
        logger.error(f"Error initializing predictor: {e}")
        
        # Try to load any available model
        try:
            available_models = ['naive_bayes', 'random_forest', 'logistic_regression']
            for model_name in available_models:
                try:
                    predictor = SpamPredictor(
                        model_name=model_name,
                        models_path=str(models_path)
                    )
                    logger.info(f"Loaded fallback model: {model_name}")
                    break
                except:
                    continue
        except Exception as fallback_error:
            logger.error(f"Failed to load any model: {fallback_error}")
            predictor = None


@app.route('/')
def index():
    """
    Serve the main web interface.
    
    Returns:
        HTML: The main application page
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Classify a single message as spam or ham.
    
    Request JSON:
        {
            "message": "Your message text here",
            "model": "logistic_regression" (optional)
        }
    
    Returns:
        JSON: {
            "prediction": "spam" or "ham",
            "confidence": 0.95,
            "model_used": "logistic_regression",
            "processing_time_ms": 12,
            "timestamp": "2025-01-17T19:00:00"
        }
    """
    global stats, prediction_history
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Missing required field: message'
            }), 400
        
        message = data['message'].strip()
        
        if not message:
            return jsonify({
                'error': 'Message cannot be empty'
            }), 400
        
        # Check if predictor is initialized
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.'
            }), 503
        
        # Optional: switch model for this prediction
        model_name = data.get('model', None)
        
        # Make prediction
        start_time = time.time()
        result = predictor.predict_message(message, model_name=model_name)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update statistics
        stats['total_predictions'] += 1
        if result.prediction == 'spam':
            stats['spam_count'] += 1
        else:
            stats['ham_count'] += 1
        
        # Add to history (keep last 100)
        prediction_record = {
            'message': message[:100] + '...' if len(message) > 100 else message,
            'prediction': result.prediction,
            'confidence': result.confidence,
            'timestamp': datetime.now().isoformat()
        }
        prediction_history.insert(0, prediction_record)
        prediction_history = prediction_history[:100]  # Keep only last 100
        
        # Prepare response
        response = {
            'prediction': result.prediction,
            'confidence': round(result.confidence, 4),
            'model_used': result.model_used,
            'processing_time_ms': round(processing_time, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction: {result.prediction} (confidence: {result.confidence:.2%})")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Classify multiple messages at once.
    
    Request JSON:
        {
            "messages": ["message1", "message2", ...],
            "model": "logistic_regression" (optional)
        }
    
    Returns:
        JSON: {
            "predictions": [
                {"message": "...", "prediction": "spam", "confidence": 0.95},
                ...
            ],
            "total_spam": 5,
            "total_ham": 3,
            "processing_time_ms": 45,
            "model_used": "logistic_regression"
        }
    """
    global stats, prediction_history
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'messages' not in data:
            return jsonify({
                'error': 'Missing required field: messages'
            }), 400
        
        messages = data['messages']
        
        if not isinstance(messages, list):
            return jsonify({
                'error': 'messages must be a list'
            }), 400
        
        if len(messages) == 0:
            return jsonify({
                'error': 'messages list cannot be empty'
            }), 400
        
        if len(messages) > 1000:
            return jsonify({
                'error': 'Maximum 1000 messages allowed per batch'
            }), 400
        
        # Check if predictor is initialized
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.'
            }), 503
        
        # Optional: switch model for this prediction
        model_name = data.get('model', None)
        
        # Make batch prediction
        start_time = time.time()
        results = predictor.predict_batch(messages, model_name=model_name)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Process results
        predictions = []
        spam_count = 0
        ham_count = 0
        
        for result in results:
            predictions.append({
                'message': result.message[:100] + '...' if len(result.message) > 100 else result.message,
                'prediction': result.prediction,
                'confidence': round(result.confidence, 4)
            })
            
            if result.prediction == 'spam':
                spam_count += 1
            else:
                ham_count += 1
        
        # Update global statistics
        stats['total_predictions'] += len(messages)
        stats['spam_count'] += spam_count
        stats['ham_count'] += ham_count
        
        # Add to history (keep last 100)
        for pred in predictions[:10]:  # Add only first 10 to history
            prediction_record = {
                'message': pred['message'],
                'prediction': pred['prediction'],
                'confidence': pred['confidence'],
                'timestamp': datetime.now().isoformat()
            }
            prediction_history.insert(0, prediction_record)
        prediction_history = prediction_history[:100]
        
        # Prepare response
        response = {
            'predictions': predictions,
            'total_messages': len(messages),
            'total_spam': spam_count,
            'total_ham': ham_count,
            'processing_time_ms': round(processing_time, 2),
            'model_used': results[0].model_used if results else 'unknown'
        }
        
        logger.info(f"Batch prediction: {len(messages)} messages, {spam_count} spam, {ham_count} ham")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in /batch-predict: {e}")
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}'
        }), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Get application statistics.
    
    Returns:
        JSON: {
            "total_predictions": 1234,
            "spam_count": 156,
            "ham_count": 1078,
            "spam_percentage": 12.6,
            "available_models": ["naive_bayes", "logistic_regression", "random_forest"],
            "current_model": "logistic_regression",
            "start_time": "2025-01-17T10:00:00",
            "uptime_hours": 2.5
        }
    """
    try:
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 503
        
        # Calculate spam percentage
        total = stats['total_predictions']
        spam_percentage = (stats['spam_count'] / total * 100) if total > 0 else 0
        
        # Calculate uptime
        start_time = datetime.fromisoformat(stats['start_time'])
        uptime_seconds = (datetime.now() - start_time).total_seconds()
        uptime_hours = uptime_seconds / 3600
        
        # Get available models
        available_models = predictor.get_available_models()
        
        response = {
            'total_predictions': stats['total_predictions'],
            'spam_count': stats['spam_count'],
            'ham_count': stats['ham_count'],
            'spam_percentage': round(spam_percentage, 2),
            'available_models': available_models,
            'current_model': predictor.model_name,
            'start_time': stats['start_time'],
            'uptime_hours': round(uptime_hours, 2)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in /stats: {e}")
        return jsonify({
            'error': f'Failed to get statistics: {str(e)}'
        }), 500


@app.route('/models', methods=['GET'])
def get_models():
    """
    Get list of available models and their metrics.
    
    Returns:
        JSON: {
            "available_models": ["naive_bayes", "logistic_regression", "random_forest"],
            "current_model": "logistic_regression",
            "metrics": {
                "naive_bayes": {"accuracy": 0.972, "f1_score": 0.935, ...},
                ...
            }
        }
    """
    try:
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 503
        
        available_models = predictor.get_available_models()
        metrics = predictor.get_model_metrics()
        
        response = {
            'available_models': available_models,
            'current_model': predictor.model_name,
            'metrics': metrics
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in /models: {e}")
        return jsonify({
            'error': f'Failed to get models: {str(e)}'
        }), 500


@app.route('/switch-model', methods=['POST'])
def switch_model():
    """
    Switch to a different model.
    
    Request JSON:
        {
            "model": "naive_bayes"
        }
    
    Returns:
        JSON: {
            "success": true,
            "current_model": "naive_bayes",
            "message": "Successfully switched to naive_bayes"
        }
    """
    try:
        if predictor is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 503
        
        data = request.get_json()
        
        if not data or 'model' not in data:
            return jsonify({
                'error': 'Missing required field: model'
            }), 400
        
        model_name = data['model']
        
        # Switch model
        predictor.load_model(model_name)
        
        logger.info(f"Switched to model: {model_name}")
        
        response = {
            'success': True,
            'current_model': predictor.model_name,
            'message': f'Successfully switched to {model_name}'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in /switch-model: {e}")
        return jsonify({
            'error': f'Failed to switch model: {str(e)}'
        }), 500


@app.route('/history', methods=['GET'])
def get_history():
    """
    Get recent prediction history.
    
    Query params:
        limit: Number of records to return (default: 10, max: 100)
    
    Returns:
        JSON: {
            "history": [
                {"message": "...", "prediction": "spam", "confidence": 0.95, "timestamp": "..."},
                ...
            ],
            "total_records": 10
        }
    """
    try:
        limit = request.args.get('limit', default=10, type=int)
        limit = min(limit, 100)  # Cap at 100
        
        history_subset = prediction_history[:limit]
        
        response = {
            'history': history_subset,
            'total_records': len(history_subset)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in /history: {e}")
        return jsonify({
            'error': f'Failed to get history: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON: {
            "status": "healthy",
            "model_loaded": true,
            "timestamp": "2025-01-17T19:00:00"
        }
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Initialize predictor on startup
    logger.info("Starting Spam Detection Web Application...")
    initialize_predictor()
    
    if predictor is None:
        logger.warning("WARNING: No model loaded. Application may not function correctly.")
    else:
        logger.info(f"Application ready with model: {predictor.model_name}")
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on http://localhost:{port}")
    logger.info("Press CTRL+C to quit")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
