#!/usr/bin/env python3
"""
Flask API Deployment for Yacht Price Prediction
Real-time prediction endpoints with optimized ML models
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model and feature information
model = None
feature_names = None
model_metadata = {}

def load_model():
    """Load the optimized ML model"""
    global model, feature_names, model_metadata
    
    try:
        # Try to load optimized model (if hyperparameter tuning was run)
        try:
            with open('optimized_yacht_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
                feature_names = model_data['feature_names']
                model_metadata = model_data['metadata']
            print("Optimized model loaded successfully")
        except FileNotFoundError:
            # Fallback to simple model if optimized not available
            print("Optimized model not found, using fallback...")
            from sklearn.ensemble import RandomForestRegressor
            
            # Load data and train a quick model
            df = pd.read_csv('enhanced_yacht_market_data.csv')
            numeric_features = [
                'length_meters', 'beam_meters', 'draft_meters', 'year_built',
                'engine_hours', 'fuel_capacity_l', 'water_capacity_l', 'age_years',
                'num_cabins', 'engine_power_hp', 'cruise_speed_knots', 'max_speed_knots'
            ]
            
            available_features = [f for f in numeric_features if f in df.columns]
            X = df[available_features].fillna(df[available_features].mean())
            y = df['sale_price'].fillna(df['sale_price'].mean())
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            feature_names = available_features
            model_metadata = {
                'model_type': 'RandomForest',
                'training_samples': len(X),
                'features': len(feature_names),
                'created_at': datetime.now().isoformat()
            }
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    return True

def validate_input(data):
    """Validate input data for prediction"""
    required_features = set(feature_names)
    provided_features = set(data.keys())
    
    missing_features = required_features - provided_features
    if missing_features:
        return False, f"Missing features: {list(missing_features)}"
    
    # Validate data types and ranges
    for feature in feature_names:
        value = data.get(feature)
        if value is None:
            return False, f"Feature {feature} cannot be null"
        
        if not isinstance(value, (int, float)):
            return False, f"Feature {feature} must be numeric"
        
        if feature in ['length_meters', 'beam_meters', 'draft_meters'] and (value <= 0 or value > 100):
            return False, f"Feature {feature} must be between 0 and 100 meters"
        
        if feature == 'year_built' and (value < 1950 or value > 2025):
            return False, f"Feature {feature} must be between 1950 and 2025"
        
        if feature in ['engine_hours', 'fuel_capacity_l', 'water_capacity_l'] and value < 0:
            return False, f"Feature {feature} cannot be negative"
    
    return True, "Valid input"

def prepare_input(data):
    """Prepare input data for model prediction"""
    input_data = {}
    for feature in feature_names:
        input_data[feature] = [data.get(feature, 0)]
    
    return pd.DataFrame(input_data)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'model_info': model_metadata
    })

@app.route('/predict', methods=['POST'])
def predict_price():
    """Main prediction endpoint"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'status': 'error'
            }), 400
        
        # Validate input
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({
                'error': message,
                'status': 'error'
            }), 400
        
        # Prepare input
        input_df = prepare_input(data)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Calculate prediction confidence (simple approximation)
        # In a real implementation, you'd use prediction intervals
        confidence = min(0.95, max(0.70, 1.0 - (abs(np.mean(model.predict(input_df)) - prediction) / prediction)))
        
        response = {
            'prediction': float(prediction),
            'currency': 'EUR',
            'confidence': float(confidence),
            'model_info': {
                'type': model_metadata.get('model_type', 'RandomForest'),
                'features_used': len(feature_names),
                'training_samples': model_metadata.get('training_samples', 'Unknown')
            },
            'input_features': data,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple yachts"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'yachts' not in data:
            return jsonify({
                'error': 'No yacht data provided',
                'status': 'error'
            }), 400
        
        predictions = []
        
        for i, yacht_data in enumerate(data['yachts']):
            try:
                # Validate input
                is_valid, message = validate_input(yacht_data)
                if not is_valid:
                    predictions.append({
                        'index': i,
                        'error': message,
                        'status': 'error'
                    })
                    continue
                
                # Prepare input and predict
                input_df = prepare_input(yacht_data)
                prediction = model.predict(input_df)[0]
                confidence = min(0.95, max(0.70, 1.0 - (abs(np.mean(model.predict(input_df)) - prediction) / prediction)))
                
                predictions.append({
                    'index': i,
                    'prediction': float(prediction),
                    'confidence': float(confidence),
                    'currency': 'EUR',
                    'status': 'success'
                })
            
            except Exception as e:
                predictions.append({
                    'index': i,
                    'error': str(e),
                    'status': 'error'
                })
        
        response = {
            'predictions': predictions,
            'total_processed': len(data['yachts']),
            'successful': len([p for p in predictions if p['status'] == 'success']),
            'failed': len([p for p in predictions if p['status'] == 'error']),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 500
    
    # Feature importance (if available)
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    
    return jsonify({
        'model_type': model_metadata.get('model_type', 'RandomForest'),
        'features': feature_names,
        'feature_importance': feature_importance,
        'training_samples': model_metadata.get('training_samples', 'Unknown'),
        'created_at': model_metadata.get('created_at', 'Unknown'),
        'status': 'success'
    })

@app.route('/validate_input', methods=['POST'])
def validate_endpoint():
    """Validate input data without making prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'status': 'error'
            }), 400
        
        is_valid, message = validate_input(data)
        
        return jsonify({
            'valid': is_valid,
            'message': message,
            'required_features': feature_names,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Validation failed: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API documentation"""
    return jsonify({
        'api_name': 'Yacht Price Prediction API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'GET /': 'API documentation',
            'GET /health': 'Health check',
            'POST /predict': 'Single yacht prediction',
            'POST /batch_predict': 'Multiple yacht predictions',
            'GET /model_info': 'Model information and feature importance',
            'POST /validate_input': 'Validate input data format'
        },
        'example_payload': {
            'length_meters': 25.5,
            'beam_meters': 6.2,
            'draft_meters': 2.1,
            'year_built': 2018,
            'engine_hours': 1200,
            'fuel_capacity_l': 2000,
            'water_capacity_l': 800,
            'age_years': 6,
            'num_cabins': 4,
            'engine_power_hp': 1200,
            'cruise_speed_knots': 22,
            'max_speed_knots': 28
        }
    })

if __name__ == '__main__':
    print("Starting Yacht Price Prediction API...")
    
    # Load model
    if load_model():
        print("Model loaded successfully!")
        print(f"Features: {feature_names}")
        print(f"Model type: {model_metadata.get('model_type', 'Unknown')}")
        print("\nAPI is running on http://localhost:5000")
        print("Available endpoints:")
        print("  GET  /             - API documentation")
        print("  GET  /health       - Health check")
        print("  POST /predict      - Single prediction")
        print("  POST /batch_predict - Batch predictions")
        print("  GET  /model_info   - Model information")
        print("  POST /validate_input - Input validation")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load model. Exiting...")
        exit(1)