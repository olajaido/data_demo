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
warnings.filterwarnings('ignore')

class RetailAnalyzer:
    def __init__(self, data_path):
        """Initialize the RetailAnalyzer with data path"""
        self.data_path = data_path
        self.df = None
        self.features_df = None
        
    def load_and_preprocess(self):
        """Load and preprocess the retail dataset"""
        try:
            # Extract bucket and key from S3 path
            print(f"Attempting to access: {self.data_path}")
            bucket = self.data_path.split('/')[2]
            key = '/'.join(self.data_path.split('/')[3:])
            print(f"Bucket: {bucket}, Key: {key}")
            
            # Initialize S3 client
            s3_client = boto3.client('s3')
            print("S3 client initialized")
            
            try:
                # Get object from S3
                print("Attempting to get object from S3...")
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                print("Successfully retrieved object from S3")
                
                # Read Excel file
                print("Reading Excel file...")
                self.df = pd.read_excel(io.BytesIO(obj['Body'].read()))
                print(f"Excel file loaded. Shape: {self.df.shape}")
                
                # Print column names for debugging
                print(f"Columns in Excel file: {self.df.columns.tolist()}")
                
                # Convert Invoice Date to datetime
                print("Converting Invoice Date...")
                self.df['Invoice Date'] = pd.to_datetime(self.df['Invoice Date'])
                
                # Create total amount column
                print("Calculating total amounts...")
                self.df['TotalAmount'] = self.df['Quantity'] * self.df['Price']
                
                # Remove cancelled orders
                print("Removing cancelled orders...")
                original_len = len(self.df)
                self.df = self.df[~self.df['Invoice'].astype(str).str.startswith('C')]
                print(f"Removed {original_len - len(self.df)} cancelled orders")
                
                # Remove rows with missing Customer ID
                print("Handling missing Customer IDs...")
                original_len = len(self.df)
                self.df = self.df.dropna(subset=['Customer ID'])
                print(f"Removed {original_len - len(self.df)} rows with missing Customer IDs")
                
                # Remove invalid quantities and prices
                print("Removing invalid quantities and prices...")
                original_len = len(self.df)
                self.df = self.df[(self.df['Quantity'] > 0) & (self.df['Price'] > 0)]
                print(f"Removed {original_len - len(self.df)} rows with invalid quantities/prices")
                
                # Rename columns
                print("Renaming columns...")
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
                print("Column renaming complete")
                
                print(f"Final dataframe shape: {self.df.shape}")
                return self.df
                
            except boto3.exceptions.Boto3Error as e:
                print(f"AWS Error: {str(e)}")
                raise Exception(f"Failed to access S3: {str(e)}")
                
            except pd.errors.EmptyDataError:
                print("Error: The Excel file is empty")
                raise Exception("The Excel file is empty")
                
            except Exception as e:
                print(f"Error processing file: {str(e)}")
                raise Exception(f"Error processing file: {str(e)}")
                
        except Exception as e:
            print(f"Error in load_and_preprocess: {str(e)}")
            return None
    
    def create_customer_features(self):
        """Create customer-level features for clustering"""
        customer_features = self.df.groupby('CustomerID').agg({
            'InvoiceNo': 'count',  # Frequency
            'TotalAmount': ['sum', 'mean'],  # Monetary
            'InvoiceDate': lambda x: (x.max() - x.min()).days  # Recency
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = ['CustomerID', 'Frequency', 'TotalSpent', 'AvgTransactionValue', 'CustomerLifespan']
        
        # Add additional features
        customer_features['AvgPurchaseFrequency'] = customer_features['Frequency'] / customer_features['CustomerLifespan']
        
        self.features_df = customer_features
        return customer_features
    
    def perform_clustering(self, n_clusters=5):
        """Perform K-means clustering on customer features"""
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
        
        return self.features_df
    
    def analyze_clusters(self):
        """Analyze and visualize clustering results"""
        cluster_summary = self.features_df.groupby('Cluster').agg({
            'Frequency': 'mean',
            'TotalSpent': 'mean',
            'AvgTransactionValue': 'mean',
            'CustomerLifespan': 'mean',
            'AvgPurchaseFrequency': 'mean',
            'CustomerID': 'count'  # Number of customers in each cluster
        }).round(2)
        
        return cluster_summary
    
    def plot_clusters(self):
        """Create visualizations for clusters"""
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
        plt.show()
    
    def generate_sales_trends(self):
        """Generate and visualize sales trends"""
        # Monthly sales trend
        monthly_sales = self.df.groupby(self.df['InvoiceDate'].dt.to_period('M')).agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        monthly_sales['InvoiceDate'] = monthly_sales['InvoiceDate'].astype(str)
        
        return monthly_sales
    
    def analyze_geographic_distribution(self):
        """Analyze sales distribution by country"""
        country_analysis = self.df.groupby('Country').agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique'
        }).round(2).sort_values('TotalAmount', ascending=False)
        
        return country_analysis
    
    def analyze_product_performance(self):
        """Analyze product performance"""
        product_analysis = self.df.groupby(['StockCode', 'Description']).agg({
            'Quantity': 'sum',
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique'
        }).round(2).sort_values('TotalAmount', ascending=False)
        
        return product_analysis

# Usage example:
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = RetailAnalyzer('s3://retail-analysis-data-demo/online_retail_II.xlsx')
    
    # Load and preprocess data
    df = analyzer.load_and_preprocess()
    
    # Create customer features
    customer_features = analyzer.create_customer_features()
    
    # Perform clustering
    clustered_customers = analyzer.perform_clustering(n_clusters=5)
    
    # Analyze clusters
    cluster_summary = analyzer.analyze_clusters()
    
    # Generate sales trends
    sales_trends = analyzer.generate_sales_trends()
    
    # Analyze geographic distribution
    geo_analysis = analyzer.analyze_geographic_distribution()
    
    # Analyze product performance
    product_analysis = analyzer.analyze_product_performance()