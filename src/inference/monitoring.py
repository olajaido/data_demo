import time
import boto3
from datetime import datetime
import logging
from logger_config import logger

class ModelMonitor:
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.namespace = 'RetailAnalysis/Inference'

    def log_latency(self, start_time, end_time, endpoint_name):
        """Log inference latency"""
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        self.put_metric('InferenceLatency', latency, endpoint_name)

    def log_prediction(self, prediction, endpoint_name):
        """Log prediction distribution"""
        self.put_metric('ClusterPrediction', prediction, endpoint_name)

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
        except Exception as e:
            logger.error(f"Error putting metric {metric_name}: {str(e)}")