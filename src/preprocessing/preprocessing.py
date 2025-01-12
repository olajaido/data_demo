import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, required=True)
    parser.add_argument('--output-data', type=str, required=True)
    return parser.parse_args()

def preprocess_data(input_path, output_path):
    try:
        logger.info(f"Reading data from {input_path}")
        df = pd.read_excel(input_path + '/online_retail_II.xlsx')
        logger.info(f"Data loaded with shape: {df.shape}")

        # Data cleaning
        logger.info("Starting data preprocessing...")
        
        # Convert InvoiceDate to datetime (corrected column name)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Create total amount column
        df['TotalAmount'] = df['Quantity'] * df['Price']
        
        # Remove cancelled orders
        df = df[~df['Invoice'].astype(str).str.startswith('C')]
        
        # Remove rows with missing Customer ID
        df = df.dropna(subset=['Customer ID'])
        
        # Remove invalid quantities or prices
        df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
        
        # Rename columns
        column_mapping = {
            'Invoice': 'InvoiceNo',
            'StockCode': 'StockCode',
            'Description': 'Description',
            'Quantity': 'Quantity',
            'Price': 'UnitPrice',
            'Customer ID': 'CustomerID',
            'Country': 'Country'
        }
        df = df.rename(columns=column_mapping)
        
        # Save processed data
        output_file = os.path.join(output_path, 'processed_data.csv')
        df.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    args = parse_args()
    preprocess_data(args.input_data, args.output_data)