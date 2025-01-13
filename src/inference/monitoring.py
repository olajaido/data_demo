# import time
# import boto3
# from datetime import datetime
# import logging
# from logger_config import logger

# class ModelMonitor:
#     def __init__(self):
#         self.cloudwatch = boto3.client('cloudwatch')
#         self.namespace = 'RetailAnalysis/Inference'

#     def log_latency(self, start_time, end_time, endpoint_name):
#         """Log inference latency"""
#         latency = (end_time - start_time) * 1000  # Convert to milliseconds
#         self.put_metric('InferenceLatency', latency, endpoint_name)

#     def log_prediction(self, prediction, endpoint_name):
#         """Log prediction distribution"""
#         self.put_metric('ClusterPrediction', prediction, endpoint_name)

#     def log_error(self, error_type, endpoint_name):
#         """Log inference errors"""
#         self.put_metric('InferenceError', 1, endpoint_name, error_type)

#     def put_metric(self, metric_name, value, endpoint_name, error_type=None):
#         """Put metric to CloudWatch"""
#         try:
#             dimensions = [
#                 {'Name': 'EndpointName', 'Value': endpoint_name}
#             ]
#             if error_type:
#                 dimensions.append({'Name': 'ErrorType', 'Value': error_type})

#             self.cloudwatch.put_metric_data(
#                 Namespace=self.namespace,
#                 MetricData=[{
#                     'MetricName': metric_name,
#                     'Value': value,
#                     'Unit': 'None',
#                     'Dimensions': dimensions,
#                     'Timestamp': datetime.utcnow()
#                 }]
#             )
#         except Exception as e:
#             logger.error(f"Error putting metric {metric_name}: {str(e)}")

import time
import boto3
from datetime import datetime
import logging
from logger_config import get_logger

# Initialize logger
logger = get_logger('ModelMonitoring')

def monitor_prediction(input_data, predictions):
    """Monitor predictions as expected by the serve script"""
    try:
        monitor = ModelMonitor()
        endpoint_name = 'retail-clustering-endpoint'  # This should match your endpoint name
        
        # Log prediction
        if isinstance(predictions, (list, tuple)):
            for pred in predictions:
                monitor.log_prediction(pred, endpoint_name)
        else:
            monitor.log_prediction(predictions, endpoint_name)
            
        return True
    except Exception as e:
        logger.error(f"Monitoring failed: {str(e)}")
        return False

class ModelMonitor:
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.namespace = 'RetailAnalysis/Inference'
        self.start_time = time.time()

    def log_latency(self, start_time=None, end_time=None, endpoint_name=None):
        """Log inference latency"""
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = time.time()
        if endpoint_name is None:
            endpoint_name = 'retail-clustering-endpoint'

        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        self.put_metric('InferenceLatency', latency, endpoint_name)

    def log_prediction(self, prediction, endpoint_name):
        """Log prediction distribution"""
        try:
            # Convert prediction to float if possible
            pred_value = float(prediction)
            self.put_metric('ClusterPrediction', pred_value, endpoint_name)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert prediction to float: {str(e)}")
            # Log as a count instead
            self.put_metric('PredictionCount', 1, endpoint_name)

    def log_error(self, error_type, endpoint_name):
        """Log inference errors"""
        self.put_metric('InferenceError', 1, endpoint_name, error_type)

    def put_metric(self, metric_name, value, endpoint_name, error_type=None):
        """Put metric to CloudWatch"""
        try:
            dimensions = [
                {'Name': 'EndpointName', 'Value': endpoint_name}
            ]
            if error_type:
                dimensions.append({'Name': 'ErrorType', 'Value': error_type})

            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[{
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': 'None',
                    'Dimensions': dimensions,
                    'Timestamp': datetime.utcnow()
                }]
            )
            logger.info(f"Successfully logged metric {metric_name}: {value}")
        except Exception as e:
            logger.error(f"Error putting metric {metric_name}: {str(e)}")
            
    def __del__(self):
        """Log final latency when monitor is destroyed"""
        try:
            self.log_latency()
        except:
            pass  # Ignore errors during cleanup