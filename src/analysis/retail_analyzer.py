# src/analysis/retail_analyzer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
import warnings
import boto3
import io
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class RetailAnalyzer:
    def __init__(self, data_path):
        """Initialize the RetailAnalyzer with data path"""
        self.data_path = data_path
        self.df = None
        self.features_df = None
        logger.info(f"RetailAnalyzer initialized with path: {data_path}")
        
    def load_and_preprocess(self):
        """Load and preprocess the retail dataset"""
        try:
            logger.info(f"Starting to load data from: {self.data_path}")
            
            # Extract bucket and key from S3 path
            bucket = self.data_path.split('/')[2]
            key = '/'.join(self.data_path.split('/')[3:])
            logger.info(f"Parsed S3 path - Bucket: {bucket}, Key: {key}")
            
            # Initialize S3 client with region
            s3_client = boto3.client('s3', region_name='eu-west-2')
            logger.info("Initialized S3 client in eu-west-2")
            
            try:
                # Get object from S3
                logger.info("Attempting to get object from S3...")
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                logger.info("Successfully retrieved object from S3")
                
                # Read Excel file
                logger.info("Reading Excel file...")
                self.df = pd.read_excel(io.BytesIO(obj['Body'].read()))
                logger.info(f"Excel file loaded. Shape: {self.df.shape}")
                logger.info(f"Columns in Excel file: {self.df.columns.tolist()}")
                
                # Data Processing steps
                logger.info("Starting data preprocessing...")
                
                # Convert Invoice Date to datetime
                self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
                logger.info("Converted dates to datetime")
                
                # Create total amount column
                self.df['TotalAmount'] = self.df['Quantity'] * self.df['Price']
                logger.info("Calculated total amounts")
                
                # Remove cancelled orders
                original_len = len(self.df)
                self.df = self.df[~self.df['Invoice'].astype(str).str.startswith('C')]
                logger.info(f"Removed {original_len - len(self.df)} cancelled orders")
                
                # Remove rows with missing Customer ID
                original_len = len(self.df)
                self.df = self.df.dropna(subset=['Customer ID'])
                logger.info(f"Removed {original_len - len(self.df)} rows with missing Customer IDs")
                
                # Remove invalid quantities and prices
                original_len = len(self.df)
                self.df = self.df[(self.df['Quantity'] > 0) & (self.df['Price'] > 0)]
                logger.info(f"Removed {original_len - len(self.df)} rows with invalid quantities/prices")
                
                # Rename columns
                self.df = self.df.rename(columns={
                    'Invoice': 'InvoiceNo',
                    'StockCode': 'StockCode',
                    'Description': 'Description',
                    'Quantity': 'Quantity',
                    'Invoice Date': 'InvoiceDate',
                    'Price': 'UnitPrice',
                    'Customer ID': 'CustomerID',
                    'Country': 'Country'
                })
                logger.info("Column renaming complete")
                
                logger.info(f"Final dataframe shape: {self.df.shape}")
                return self.df
                
            except boto3.exceptions.Boto3Error as e:
                logger.error(f"AWS Error: {str(e)}", exc_info=True)
                raise Exception(f"Failed to access S3: {str(e)}")
                
            except pd.errors.EmptyDataError:
                logger.error("Empty Excel file error", exc_info=True)
                raise Exception("The Excel file is empty")
                
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}", exc_info=True)
                raise Exception(f"Error processing file: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in load_and_preprocess: {str(e)}", exc_info=True)
            return None
    
    def create_customer_features(self):
        """Create customer-level features for clustering"""
        try:
            logger.info("Creating customer features...")
            customer_features = self.df.groupby('CustomerID').agg({
                'InvoiceNo': 'count',  # Frequency
                'TotalAmount': ['sum', 'mean'],  # Monetary
                'InvoiceDate': lambda x: (x.max() - x.min()).days  # Recency
            }).reset_index()
            
            # Flatten column names
            customer_features.columns = ['CustomerID', 'Frequency', 'TotalSpent', 'AvgTransactionValue', 'CustomerLifespan']
            
            # Handle zero values in CustomerLifespan to avoid division by zero
            customer_features['CustomerLifespan'] = customer_features['CustomerLifespan'].replace(0, 1)
            
            # Add additional features with safeguard against infinity
            customer_features['AvgPurchaseFrequency'] = (customer_features['Frequency'] / customer_features['CustomerLifespan']).clip(upper=1e6)
            
            # Replace any infinity values with 0
            customer_features = customer_features.replace([np.inf, -np.inf], 0)
            
            self.features_df = customer_features
            logger.info(f"Customer features created. Shape: {customer_features.shape}")
            return customer_features
        except Exception as e:
            logger.error(f"Error creating customer features: {str(e)}", exc_info=True)
            raise
    
    def perform_clustering(self, n_clusters=5):
        """Perform K-means clustering on customer features"""
        try:
            logger.info(f"Starting clustering with {n_clusters} clusters...")
            # Prepare features for clustering
            features_for_clustering = self.features_df.drop('CustomerID', axis=1)
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_for_clustering)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            # Add cluster labels to features dataframe
            self.features_df['Cluster'] = clusters
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(scaled_features, clusters)
            logger.info(f"Clustering complete. Silhouette score: {silhouette_avg:.3f}")
            
            return self.features_df
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}", exc_info=True)
            raise
    
    def analyze_clusters(self):
        """Analyze clustering results"""
        try:
            logger.info("Analyzing clusters...")
            cluster_summary = self.features_df.groupby('Cluster').agg({
                'Frequency': 'mean',
                'TotalSpent': 'mean',
                'AvgTransactionValue': 'mean',
                'CustomerLifespan': 'mean',
                'AvgPurchaseFrequency': 'mean',
                'CustomerID': 'count'  # Number of customers in cluster
            }).round(2)
            
            logger.info("Cluster analysis complete")
            return cluster_summary
        except Exception as e:
            logger.error(f"Error analyzing clusters: {str(e)}", exc_info=True)
            raise
    
    def plot_clusters(self):
        """Create cluster visualizations"""
        try:
            logger.info("Creating cluster visualizations...")
            # Perform PCA for visualization
            pca = PCA(n_components=2)
            features_for_pca = self.features_df.drop(['CustomerID', 'Cluster'], axis=1)
            scaled_features = StandardScaler().fit_transform(features_for_pca)
            pca_result = pca.fit_transform(scaled_features)
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                                c=self.features_df['Cluster'], cmap='viridis')
            plt.title('Customer Segments Visualization (PCA)')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.colorbar(scatter)
            
            logger.info("Cluster visualization complete")
            return plt
        except Exception as e:
            logger.error(f"Error creating cluster visualization: {str(e)}", exc_info=True)
            raise
    
    def generate_sales_trends(self):
        """Generate sales trends analysis"""
        try:
            logger.info("Generating sales trends...")
            # Monthly sales trend
            monthly_sales = self.df.groupby(self.df['InvoiceDate'].dt.to_period('M')).agg({
                'TotalAmount': 'sum',
                'InvoiceNo': 'nunique',
                'CustomerID': 'nunique'
            }).reset_index()
            monthly_sales['InvoiceDate'] = monthly_sales['InvoiceDate'].astype(str)
            
            logger.info("Sales trends generated successfully")
            return monthly_sales
        except Exception as e:
            logger.error(f"Error generating sales trends: {str(e)}", exc_info=True)
            raise
    
    def analyze_geographic_distribution(self):
        """Analyze sales distribution by country"""
        try:
            logger.info("Analyzing geographic distribution...")
            country_analysis = self.df.groupby('Country').agg({
                'TotalAmount': 'sum',
                'InvoiceNo': 'nunique',
                'CustomerID': 'nunique'
            }).round(2).sort_values('TotalAmount', ascending=False)
            
            logger.info("Geographic analysis complete")
            return country_analysis
        except Exception as e:
            logger.error(f"Error in geographic analysis: {str(e)}", exc_info=True)
            raise
    
    def analyze_product_performance(self):
        """Analyze product performance"""
        try:
            logger.info("Analyzing product performance...")
            product_analysis = self.df.groupby(['StockCode', 'Description']).agg({
                'Quantity': 'sum',
                'TotalAmount': 'sum',
                'InvoiceNo': 'nunique'
            }).round(2).sort_values('TotalAmount', ascending=False)
            
            logger.info("Product performance analysis complete")
            return product_analysis
        except Exception as e:
            logger.error(f"Error in product analysis: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    try:
        logger.info("Starting RetailAnalyzer test")
        # Initialize analyzer
        analyzer = RetailAnalyzer('s3://retail-analysis-data-demo/online_retail_II.xlsx')
        
        # Load and preprocess data
        df = analyzer.load_and_preprocess()
        if df is not None:
            logger.info("Test successful")
        else:
            logger.error("Test failed - data loading returned None")
    except Exception as e:
        logger.error("Test failed with exception", exc_info=True)