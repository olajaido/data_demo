import os
import json
import joblib
import numpy as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def model_fn(model_dir):
    """Load the model and scaler"""
    logger.info("Loading model.")
    kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    return {'model': kmeans, 'scaler': scaler}

def input_fn(request_body, request_content_type):
    """Parse input data"""
    logger.info("Processing input data.")
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        features = pd.DataFrame([data])
        return features
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction"""
    logger.info("Making prediction.")
    scaled_features = model['scaler'].transform(input_data)
    predictions = model['model'].predict(scaled_features)
    return predictions

def output_fn(prediction, accept):
    """Format prediction output"""
    logger.info("Processing prediction output.")
    if accept == 'application/json':
        return json.dumps({'cluster': int(prediction[0])})
    raise ValueError(f"Unsupported accept type: {accept}")