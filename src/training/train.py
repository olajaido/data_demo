import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-model', type=str, default='/opt/ml/model')
    parser.add_argument('--n-clusters', type=int, default=5)
    return parser.parse_args()

def create_features(df):
    """Create customer features for clustering"""
    customer_features = df.groupby('CustomerID').agg({
        'InvoiceNo': 'count',  # Frequency
        'TotalAmount': ['sum', 'mean'],  # Monetary
        'InvoiceDate': lambda x: (x.max() - x.min()).days  # Recency
    }).reset_index()
    
    # Flatten column names
    customer_features.columns = ['CustomerID', 'Frequency', 'TotalSpent', 'AvgTransactionValue', 'CustomerLifespan']
    
    # Add additional features
    customer_features['AvgPurchaseFrequency'] = customer_features['Frequency'] / customer_features['CustomerLifespan'].clip(lower=1)
    
    return customer_features

def train_model(input_path, output_path, n_clusters):
    try:
        # Read processed data
        data_file = os.path.join(input_path, 'processed_data.csv')
        logger.info(f"Reading processed data from {data_file}")
        df = pd.read_csv(data_file)
        
        # Create features
        logger.info("Creating customer features")
        features_df = create_features(df)
        
        # Prepare features for clustering
        features_for_clustering = features_df.drop('CustomerID', axis=1)
        
        # Scale features
        logger.info("Scaling features")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_for_clustering)
        
        # Train KMeans
        logger.info(f"Training KMeans with {n_clusters} clusters")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(scaled_features)
        
        # Save models
        logger.info(f"Saving models to {output_path}")
        joblib.dump(kmeans, os.path.join(output_path, 'kmeans_model.joblib'))
        joblib.dump(scaler, os.path.join(output_path, 'scaler.joblib'))
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise

if __name__ == '__main__':
    args = parse_args()
    train_model(args.input_data, args.output_model, args.n_clusters)