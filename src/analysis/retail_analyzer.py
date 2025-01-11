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
warnings.filterwarnings('ignore')

class RetailAnalyzer:
    def __init__(self, data_path):
        """Initialize the RetailAnalyzer with data path"""
        self.data_path = data_path
        self.df = None
        self.features_df = None
        
    def load_and_preprocess(self):
        """Load and preprocess the retail dataset"""
        # Load data
        self.df = pd.read_csv(self.data_path, encoding='ISO-8859-1')
        
        # Convert InvoiceDate to datetime
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        
        # Create total amount column
        self.df['TotalAmount'] = self.df['Quantity'] * self.df['UnitPrice']
        
        # Remove cancelled orders (those starting with 'C')
        self.df = self.df[~self.df['InvoiceNo'].astype(str).str.startswith('C')]
        
        # Remove rows with missing CustomerID
        self.df = self.df.dropna(subset=['CustomerID'])
        
        # Remove entries with negative quantities or prices
        self.df = self.df[(self.df['Quantity'] > 0) & (self.df['UnitPrice'] > 0)]
        
        return self.df
    
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
    analyzer = RetailAnalyzer('path_to_your_data.csv')
    
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