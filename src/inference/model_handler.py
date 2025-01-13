# import os
# import joblib
# import pandas as pd
# import numpy as np
# from logger_config import logger
# from monitoring import ModelMonitor

# class ModelHandler:
#     def __init__(self):
#         self.model = None
#         self.scaler = None
#         self.monitor = ModelMonitor()
        
#     def load(self, model_dir):
#         """Load model artifacts with error handling"""
#         try:
#             model_path = os.path.join(model_dir, 'kmeans_model.joblib')
#             scaler_path = os.path.join(model_dir, 'scaler.joblib')
            
#             if not os.path.exists(model_path):
#                 raise FileNotFoundError(f"Model file not found at {model_path}")
#             if not os.path.exists(scaler_path):
#                 raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            
#             self.model = joblib.load(model_path)
#             self.scaler = joblib.load(scaler_path)
#             logger.info("Model and scaler loaded successfully")
#             return {'model': self.model, 'scaler': self.scaler}
            
#         except Exception as e:
#             logger.error(f"Error loading model: {str(e)}")
#             self.monitor.log_error('ModelLoadError', 'retail-clustering-endpoint')
#             raise
        
#     def preprocess(self, input_data):
#         """Preprocess with validation"""
#         try:
#             # Validate input
#             required_features = ['Frequency', 'TotalSpent', 'AvgTransactionValue', 
#                                'CustomerLifespan', 'AvgPurchaseFrequency']
            
#             if isinstance(input_data, dict):
#                 missing_features = [f for f in required_features if f not in input_data]
#                 if missing_features:
#                     raise ValueError(f"Missing required features: {missing_features}")
#                 input_data = pd.DataFrame([input_data])
            
#             # Validate data types
#             for feature in required_features:
#                 if not np.issubdtype(input_data[feature].dtype, np.number):
#                     raise ValueError(f"Feature {feature} must be numeric")
            
#             logger.info("Input data preprocessed successfully")
#             return input_data
            
#         except Exception as e:
#             logger.error(f"Error preprocessing data: {str(e)}")
#             self.monitor.log_error('PreprocessError', 'retail-clustering-endpoint')
#             raise
        
#     def predict(self, input_data, model):
#         """Make predictions with monitoring"""
#         try:
#             start_time = time.time()
            
#             scaled_data = model['scaler'].transform(input_data)
#             predictions = model['model'].predict(scaled_data)
            
#             end_time = time.time()
#             self.monitor.log_latency(start_time, end_time, 'retail-clustering-endpoint')
            
#             prediction = predictions[0] if len(predictions) == 1 else predictions
#             self.monitor.log_prediction(prediction, 'retail-clustering-endpoint')
            
#             logger.info(f"Prediction made successfully: {prediction}")
#             return prediction
            
#         except Exception as e:
#             logger.error(f"Error making prediction: {str(e)}")
#             self.monitor.log_error('PredictionError', 'retail-clustering-endpoint')
#             raise


import os
import time
import joblib
import pandas as pd
import numpy as np
from logger_config import get_logger
from monitoring import ModelMonitor

# Initialize logger
logger = get_logger('ModelHandler')

class ModelHandler:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.monitor = ModelMonitor()
        self._load_model()
        
    def _load_model(self):
        """Load model artifacts from SageMaker model directory"""
        try:
            model_dir = '/opt/ml/model'  # SageMaker's default model directory
            model_path = os.path.join(model_dir, 'kmeans_model.joblib')
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Model and scaler loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.monitor.log_error('ModelLoadError', 'retail-clustering-endpoint')
            raise
        
    def preprocess(self, input_data):
        """Preprocess with validation"""
        try:
            required_features = ['Frequency', 'TotalSpent', 'AvgTransactionValue', 
                               'CustomerLifespan', 'AvgPurchaseFrequency']
            
            # Handle both single record and multiple records
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                df = pd.DataFrame(input_data)
            else:
                df = pd.DataFrame(input_data)
            
            # Validate required features
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Validate data types
            for feature in required_features:
                if not np.issubdtype(df[feature].dtype, np.number):
                    raise ValueError(f"Feature {feature} must be numeric")
            
            logger.info(f"Input data preprocessed successfully. Shape: {df.shape}")
            return df[required_features]  # Ensure correct feature order
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            self.monitor.log_error('PreprocessError', 'retail-clustering-endpoint')
            raise
        
    def predict(self, input_data):
        """Make predictions with monitoring"""
        try:
            start_time = time.time()
            
            # Preprocess the input
            processed_data = self.preprocess(input_data)
            
            # Scale the data
            scaled_data = self.scaler.transform(processed_data)
            
            # Make predictions
            predictions = self.model.predict(scaled_data)
            
            # Log metrics
            end_time = time.time()
            self.monitor.log_latency(start_time, end_time, 'retail-clustering-endpoint')
            
            # Handle single vs multiple predictions
            result = predictions[0] if len(predictions) == 1 else predictions.tolist()
            
            # Log prediction
            if isinstance(result, list):
                for pred in result:
                    self.monitor.log_prediction(pred, 'retail-clustering-endpoint')
            else:
                self.monitor.log_prediction(result, 'retail-clustering-endpoint')
            
            logger.info(f"Prediction made successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            self.monitor.log_error('PredictionError', 'retail-clustering-endpoint')
            raise

    def ping(self):
        """Health check for the model"""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not loaded")
        return True