# # src/analysis/retail_analyzer.py
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.decomposition import PCA
# from sklearn.cluster import AgglomerativeClustering
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.metrics import silhouette_score
# import warnings
# import boto3
# import io
# import logging

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# warnings.filterwarnings('ignore')

# class RetailAnalyzer:
#     def __init__(self, data_path):
#         """Initialize the RetailAnalyzer with data path"""
#         self.data_path = data_path
#         self.df = None
#         self.features_df = None
#         logger.info(f"RetailAnalyzer initialized with path: {data_path}")
        
#     def load_and_preprocess(self):
#         """Load and preprocess the retail dataset"""
#         try:
#             logger.info(f"Starting to load data from: {self.data_path}")
            
#             # Extract bucket and key from S3 path
#             bucket = self.data_path.split('/')[2]
#             key = '/'.join(self.data_path.split('/')[3:])
#             logger.info(f"Parsed S3 path - Bucket: {bucket}, Key: {key}")
            
#             # Initialize S3 client with region
#             s3_client = boto3.client('s3', region_name='eu-west-2')
#             logger.info("Initialized S3 client in eu-west-2")
            
#             try:
#                 # Get object from S3
#                 logger.info("Attempting to get object from S3...")
#                 obj = s3_client.get_object(Bucket=bucket, Key=key)
#                 logger.info("Successfully retrieved object from S3")
                
#                 # Read Excel file
#                 logger.info("Reading Excel file...")
#                 self.df = pd.read_excel(io.BytesIO(obj['Body'].read()))
#                 logger.info(f"Excel file loaded. Shape: {self.df.shape}")
#                 logger.info(f"Columns in Excel file: {self.df.columns.tolist()}")
                
#                 # Data Processing steps
#                 logger.info("Starting data preprocessing...")
                
#                 # Convert Invoice Date to datetime
#                 self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
#                 logger.info("Converted dates to datetime")
                
#                 # Create total amount column
#                 self.df['TotalAmount'] = self.df['Quantity'] * self.df['Price']
#                 logger.info("Calculated total amounts")
                
#                 # Remove cancelled orders
#                 original_len = len(self.df)
#                 self.df = self.df[~self.df['Invoice'].astype(str).str.startswith('C')]
#                 logger.info(f"Removed {original_len - len(self.df)} cancelled orders")
                
#                 # Remove rows with missing Customer ID
#                 original_len = len(self.df)
#                 self.df = self.df.dropna(subset=['Customer ID'])
#                 logger.info(f"Removed {original_len - len(self.df)} rows with missing Customer IDs")
                
#                 # Remove invalid quantities and prices
#                 original_len = len(self.df)
#                 self.df = self.df[(self.df['Quantity'] > 0) & (self.df['Price'] > 0)]
#                 logger.info(f"Removed {original_len - len(self.df)} rows with invalid quantities/prices")
                
#                 # Rename columns
#                 self.df = self.df.rename(columns={
#                     'Invoice': 'InvoiceNo',
#                     'StockCode': 'StockCode',
#                     'Description': 'Description',
#                     'Quantity': 'Quantity',
#                     'Invoice Date': 'InvoiceDate',
#                     'Price': 'UnitPrice',
#                     'Customer ID': 'CustomerID',
#                     'Country': 'Country'
#                 })
#                 logger.info("Column renaming complete")
                
#                 logger.info(f"Final dataframe shape: {self.df.shape}")
#                 return self.df
                
#             except boto3.exceptions.Boto3Error as e:
#                 logger.error(f"AWS Error: {str(e)}", exc_info=True)
#                 raise Exception(f"Failed to access S3: {str(e)}")
                
#             except pd.errors.EmptyDataError:
#                 logger.error("Empty Excel file error", exc_info=True)
#                 raise Exception("The Excel file is empty")
                
#             except Exception as e:
#                 logger.error(f"Error processing file: {str(e)}", exc_info=True)
#                 raise Exception(f"Error processing file: {str(e)}")
                
#         except Exception as e:
#             logger.error(f"Error in load_and_preprocess: {str(e)}", exc_info=True)
#             return None
    
#     def create_customer_features(self):
#         """Create customer-level features for clustering"""
#         try:
#             logger.info("Creating customer features...")
#             customer_features = self.df.groupby('CustomerID').agg({
#                 'InvoiceNo': 'count',  # Frequency
#                 'TotalAmount': ['sum', 'mean'],  # Monetary
#                 'InvoiceDate': lambda x: (x.max() - x.min()).days  # Recency
#             }).reset_index()
            
#             # Flatten column names
#             customer_features.columns = ['CustomerID', 'Frequency', 'TotalSpent', 'AvgTransactionValue', 'CustomerLifespan']
            
#             # Handle zero values in CustomerLifespan to avoid division by zero
#             customer_features['CustomerLifespan'] = customer_features['CustomerLifespan'].replace(0, 1)
            
#             # Add additional features with safeguard against infinity
#             customer_features['AvgPurchaseFrequency'] = (customer_features['Frequency'] / customer_features['CustomerLifespan']).clip(upper=1e6)
            
#             # Replace any infinity values with 0
#             customer_features = customer_features.replace([np.inf, -np.inf], 0)
            
#             self.features_df = customer_features
#             logger.info(f"Customer features created. Shape: {customer_features.shape}")
#             return customer_features
#         except Exception as e:
#             logger.error(f"Error creating customer features: {str(e)}", exc_info=True)
#             raise
    
#     def perform_clustering(self, n_clusters=5):
#         """Perform K-means clustering on customer features"""
#         try:
#             logger.info(f"Starting clustering with {n_clusters} clusters...")
#             # Prepare features for clustering
#             features_for_clustering = self.features_df.drop('CustomerID', axis=1)
            
#             # Scale features
#             scaler = StandardScaler()
#             scaled_features = scaler.fit_transform(features_for_clustering)
            
#             # Perform K-means clustering
#             kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#             clusters = kmeans.fit_predict(scaled_features)
            
#             # Add cluster labels to features dataframe
#             self.features_df['Cluster'] = clusters
            
#             # Calculate silhouette score
#             silhouette_avg = silhouette_score(scaled_features, clusters)
#             logger.info(f"Clustering complete. Silhouette score: {silhouette_avg:.3f}")
            
#             return self.features_df
#         except Exception as e:
#             logger.error(f"Error in clustering: {str(e)}", exc_info=True)
#             raise
    
#     def analyze_clusters(self):
#         """Analyze clustering results"""
#         try:
#             logger.info("Analyzing clusters...")
#             cluster_summary = self.features_df.groupby('Cluster').agg({
#                 'Frequency': 'mean',
#                 'TotalSpent': 'mean',
#                 'AvgTransactionValue': 'mean',
#                 'CustomerLifespan': 'mean',
#                 'AvgPurchaseFrequency': 'mean',
#                 'CustomerID': 'count'  # Number of customers in cluster
#             }).round(2)
            
#             logger.info("Cluster analysis complete")
#             return cluster_summary
#         except Exception as e:
#             logger.error(f"Error analyzing clusters: {str(e)}", exc_info=True)
#             raise
    
#     def plot_clusters(self):
#         """Create cluster visualizations"""
#         try:
#             logger.info("Creating cluster visualizations...")
#             # Perform PCA for visualization
#             pca = PCA(n_components=2)
#             features_for_pca = self.features_df.drop(['CustomerID', 'Cluster'], axis=1)
#             scaled_features = StandardScaler().fit_transform(features_for_pca)
#             pca_result = pca.fit_transform(scaled_features)
            
#             # Create scatter plot
#             plt.figure(figsize=(10, 8))
#             scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
#                                 c=self.features_df['Cluster'], cmap='viridis')
#             plt.title('Customer Segments Visualization (PCA)')
#             plt.xlabel('First Principal Component')
#             plt.ylabel('Second Principal Component')
#             plt.colorbar(scatter)
            
#             logger.info("Cluster visualization complete")
#             return plt
#         except Exception as e:
#             logger.error(f"Error creating cluster visualization: {str(e)}", exc_info=True)
#             raise
    
#     def generate_sales_trends(self):
#         """Generate sales trends analysis"""
#         try:
#             logger.info("Generating sales trends...")
#             # Monthly sales trend
#             monthly_sales = self.df.groupby(self.df['InvoiceDate'].dt.to_period('M')).agg({
#                 'TotalAmount': 'sum',
#                 'InvoiceNo': 'nunique',
#                 'CustomerID': 'nunique'
#             }).reset_index()
#             monthly_sales['InvoiceDate'] = monthly_sales['InvoiceDate'].astype(str)
            
#             logger.info("Sales trends generated successfully")
#             return monthly_sales
#         except Exception as e:
#             logger.error(f"Error generating sales trends: {str(e)}", exc_info=True)
#             raise
    
#     def analyze_geographic_distribution(self):
#         """Analyze sales distribution by country"""
#         try:
#             logger.info("Analyzing geographic distribution...")
#             country_analysis = self.df.groupby('Country').agg({
#                 'TotalAmount': 'sum',
#                 'InvoiceNo': 'nunique',
#                 'CustomerID': 'nunique'
#             }).round(2).sort_values('TotalAmount', ascending=False)
            
#             logger.info("Geographic analysis complete")
#             return country_analysis
#         except Exception as e:
#             logger.error(f"Error in geographic analysis: {str(e)}", exc_info=True)
#             raise
    
#     def analyze_product_performance(self):
#         """Analyze product performance"""
#         try:
#             logger.info("Analyzing product performance...")
#             # Convert StockCode to string type first
#             self.df['StockCode'] = self.df['StockCode'].astype(str)
            
#             product_analysis = self.df.groupby(['StockCode', 'Description']).agg({
#                 'Quantity': 'sum',
#                 'TotalAmount': 'sum',
#                 'InvoiceNo': 'nunique'
#             }).round(2).sort_values('TotalAmount', ascending=False)
            
#             logger.info("Product performance analysis complete")
#             return product_analysis
#         except Exception as e:
#             logger.error(f"Error in product analysis: {str(e)}", exc_info=True)
#             raise
#     def compare_clustering_algorithms(self, max_clusters=10):
#         """Compare different clustering algorithms"""
#         try:
#             logger.info("Comparing clustering algorithms...")
#             features_for_clustering = self.features_df.drop('CustomerID', axis=1)
#             scaled_features = StandardScaler().fit_transform(features_for_clustering)
            
#             results = {
#                 'kmeans': [],
#                 'hierarchical': [],
#                 'dbscan': []
#             }
            
#             # K-means with different k values
#             logger.info("Evaluating K-means with different cluster numbers...")
#             for k in range(2, max_clusters + 1):
#                 kmeans = KMeans(n_clusters=k, random_state=42)
#                 labels = kmeans.fit_predict(scaled_features)
#                 score = silhouette_score(scaled_features, labels)
#                 results['kmeans'].append({
#                     'n_clusters': k,
#                     'silhouette_score': score,
#                     'labels': labels
#                 })
            
#             # Hierarchical Clustering
#             logger.info("Evaluating Hierarchical Clustering...")
#             for k in range(2, max_clusters + 1):
#                 hierarchical = AgglomerativeClustering(n_clusters=k)
#                 labels = hierarchical.fit_predict(scaled_features)
#                 score = silhouette_score(scaled_features, labels)
#                 results['hierarchical'].append({
#                     'n_clusters': k,
#                     'silhouette_score': score,
#                     'labels': labels
#                 })
            
#             # DBSCAN with different parameters
#             logger.info("Evaluating DBSCAN...")
#             eps_values = [0.3, 0.5, 0.7, 1.0]
#             min_samples_values = [5, 10, 15]
            
#             for eps in eps_values:
#                 for min_samples in min_samples_values:
#                     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#                     labels = dbscan.fit_predict(scaled_features)
#                     if len(np.unique(labels[labels >= 0])) > 1:  # Check if more than one cluster
#                         score = silhouette_score(scaled_features, labels)
#                         results['dbscan'].append({
#                             'eps': eps,
#                             'min_samples': min_samples,
#                             'n_clusters': len(np.unique(labels[labels >= 0])),
#                             'silhouette_score': score,
#                             'labels': labels
#                         })
            
#             logger.info("Clustering comparison complete")
#             return results
#         except Exception as e:
#             logger.error(f"Error in clustering comparison: {str(e)}", exc_info=True)
#             raise

#     def analyze_seasonal_patterns(self):
#         """Analyze seasonal patterns in sales and customer behavior"""
#         try:
#             logger.info("Analyzing seasonal patterns...")
            
#             # Add time-based features
#             self.df['Year'] = self.df['InvoiceDate'].dt.year
#             self.df['Month'] = self.df['InvoiceDate'].dt.month
#             self.df['DayOfWeek'] = self.df['InvoiceDate'].dt.dayofweek
#             self.df['Hour'] = self.df['InvoiceDate'].dt.hour
            
#             # Define seasons
#             self.df['Season'] = pd.cut(self.df['Month'], 
#                                      bins=[0, 3, 6, 9, 12],
#                                      labels=['Winter', 'Spring', 'Summer', 'Fall'],
#                                      include_lowest=True)
            
#             # Seasonal analysis
#             seasonal_patterns = {
#                 'seasonal_sales': self.df.groupby('Season').agg({
#                     'TotalAmount': 'sum',
#                     'InvoiceNo': 'nunique',
#                     'CustomerID': 'nunique',
#                     'Quantity': 'sum'
#                 }).round(2),
                
#                 'monthly_sales': self.df.groupby(['Year', 'Month']).agg({
#                     'TotalAmount': 'sum',
#                     'InvoiceNo': 'nunique',
#                     'CustomerID': 'nunique'
#                 }).round(2),
                
#                 'daily_patterns': self.df.groupby('DayOfWeek').agg({
#                     'TotalAmount': 'sum',
#                     'InvoiceNo': 'nunique',
#                     'CustomerID': 'nunique'
#                 }).round(2),
                
#                 'hourly_patterns': self.df.groupby('Hour').agg({
#                     'TotalAmount': 'sum',
#                     'InvoiceNo': 'nunique',
#                     'CustomerID': 'nunique'
#                 }).round(2)
#             }
            
#             # Calculate growth rates
#             yearly_sales = self.df.groupby('Year')['TotalAmount'].sum()
#             seasonal_patterns['year_over_year_growth'] = (
#                 (yearly_sales - yearly_sales.shift(1)) / yearly_sales.shift(1) * 100
#             ).round(2)
            
#             logger.info("Seasonal analysis complete")
#             return seasonal_patterns
#         except Exception as e:
#             logger.error(f"Error in seasonal analysis: {str(e)}", exc_info=True)
#             raise

#     def analyze_customer_behavior(self):
#         """Detailed analysis of customer behavior patterns"""
#         try:
#             logger.info("Analyzing customer behavior patterns...")
            
#             # RFM Analysis
#             now = self.df['InvoiceDate'].max()
            
#             rfm = self.df.groupby('CustomerID').agg({
#                 'InvoiceDate': lambda x: (now - x.max()).days,  # Recency
#                 'InvoiceNo': 'count',  # Frequency
#                 'TotalAmount': 'sum'  # Monetary
#             }).reset_index()
            
#             rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
            
#             # Create RFM segments
#             r_labels = range(4, 0, -1)
#             r_quartiles = pd.qcut(rfm['Recency'], q=4, labels=r_labels)
#             f_labels = range(1, 5)
#             f_quartiles = pd.qcut(rfm['Frequency'], q=4, labels=f_labels)
#             m_labels = range(1, 5)
#             m_quartiles = pd.qcut(rfm['Monetary'], q=4, labels=m_labels)
            
#             rfm['R'] = r_quartiles
#             rfm['F'] = f_quartiles
#             rfm['M'] = m_quartiles
            
#             # Calculate RFM Score
#             rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
            
#             # Customer value segments
#             def segment_customers(row):
#                 if row['R'] == 4 and row['F'] == 4 and row['M'] == 4:
#                     return 'Best Customers'
#                 elif row['R'] == 4 and row['F'] >= 3 and row['M'] >= 3:
#                     return 'Loyal Customers'
#                 elif row['R'] >= 3 and row['F'] >= 3 and row['M'] >= 3:
#                     return 'Good Customers'
#                 elif row['R'] >= 2 and row['F'] >= 2 and row['M'] >= 2:
#                     return 'Average Customers'
#                 else:
#                     return 'Lost Customers'
            
#             rfm['Customer_Segment'] = rfm.apply(segment_customers, axis=1)
            
#             # Purchase patterns
#             purchase_patterns = {
#                 'rfm_analysis': rfm,
#                 'avg_order_value': self.df.groupby('CustomerID')['TotalAmount'].mean().describe(),
#                 'purchase_frequency': self.df.groupby('CustomerID')['InvoiceNo'].count().describe(),
#                 'customer_lifetime_value': (
#                     self.df.groupby('CustomerID')['TotalAmount'].sum() * 
#                     self.df.groupby('CustomerID')['InvoiceNo'].count()
#                 ).describe()
#             }
            
#             logger.info("Customer behavior analysis complete")
#             return purchase_patterns
#         except Exception as e:
#             logger.error(f"Error in customer behavior analysis: {str(e)}", exc_info=True)
#             raise

#     def create_enhanced_visualizations(self):
#         """Create enhanced visualizations for the analysis"""
#         try:
#             logger.info("Creating enhanced visualizations...")
            
#             visualizations = {}
            
#             # Customer Segmentation Visualization
#             pca = PCA(n_components=2)
#             features_for_pca = self.features_df.drop(['CustomerID', 'Cluster'], axis=1)
#             scaled_features = StandardScaler().fit_transform(features_for_pca)
#             pca_result = pca.fit_transform(scaled_features)
            
#             viz_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
#             viz_df['Cluster'] = self.features_df['Cluster']
#             viz_df['TotalSpent'] = self.features_df['TotalSpent']
#             viz_df['Frequency'] = self.features_df['Frequency']
            
#             visualizations['cluster_viz'] = viz_df
            
#             # Sales Trends
#             daily_sales = self.df.groupby('InvoiceDate').agg({
#                 'TotalAmount': 'sum',
#                 'InvoiceNo': 'nunique',
#                 'CustomerID': 'nunique'
#             }).reset_index()
            
#             visualizations['sales_trends'] = daily_sales
            
#             # Product Performance
#             product_metrics = self.analyze_product_performance()
#             visualizations['product_performance'] = product_metrics
            
#             logger.info("Enhanced visualizations created")
#             return visualizations
#         except Exception as e:
#             logger.error(f"Error creating enhanced visualizations: {str(e)}", exc_info=True)
#             raise    

# if __name__ == "__main__":
#     try:
#         logger.info("Starting RetailAnalyzer test")
#         # Initialize analyzer
#         analyzer = RetailAnalyzer('s3://retail-analysis-data-demo/online_retail_II.xlsx')
        
#         # Load and preprocess data
#         df = analyzer.load_and_preprocess()
#         if df is not None:
#             logger.info("Test successful")
#         else:
#             logger.error("Test failed - data loading returned None")
#     except Exception as e:
#         logger.error("Test failed with exception", exc_info=True)