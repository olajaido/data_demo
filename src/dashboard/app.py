# # src/dashboard/app.py
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import sys
# import os
# from datetime import datetime
# import numpy as np
# import logging

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Add the analysis directory to the Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'analysis'))
# from retail_analyzer import RetailAnalyzer

# def load_data():
#     """Load and process data using RetailAnalyzer"""
#     try:
#         logger.info("Starting data loading process...")
#         analyzer = RetailAnalyzer('s3://retail-analysis-data-demo/online_retail_II.xlsx')
#         logger.info("RetailAnalyzer initialized, attempting to load data...")
#         df = analyzer.load_and_preprocess()
#         if df is None:
#             logger.error("Data loading failed - DataFrame is None")
#             raise Exception("Failed to load data")
#         logger.info(f"Data loaded successfully. Shape: {df.shape}")
#         return analyzer, df
#     except Exception as e:
#         logger.error(f"Error in load_data: {str(e)}", exc_info=True)
#         raise

# def create_time_series_analysis(df):
#     """Create detailed time series analysis"""
#     try:
#         logger.info("Starting time series analysis...")
#         # Monthly trends
#         monthly = df.groupby(df['InvoiceDate'].dt.to_period('M')).agg({
#             'TotalAmount': 'sum',
#             'InvoiceNo': 'nunique',
#             'CustomerID': 'nunique'
#         }).reset_index()
#         monthly['InvoiceDate'] = monthly['InvoiceDate'].astype(str)
        
#         # Quarterly trends
#         df['Quarter'] = df['InvoiceDate'].dt.to_period('Q').astype(str)
#         quarterly = df.groupby('Quarter').agg({
#             'TotalAmount': 'sum',
#             'InvoiceNo': 'nunique',
#             'CustomerID': 'nunique'
#         }).reset_index()
        
#         # Year-over-Year comparison
#         df['Year'] = df['InvoiceDate'].dt.year
#         yearly = df.groupby('Year').agg({
#             'TotalAmount': 'sum',
#             'InvoiceNo': 'nunique',
#             'CustomerID': 'nunique'
#         }).reset_index()
        
#         logger.info("Time series analysis completed successfully")
#         return monthly, quarterly, yearly
#     except Exception as e:
#         logger.error(f"Error in time series analysis: {str(e)}", exc_info=True)
#         raise

# def create_geographic_analysis(df):
#     """Create geographic analysis visualizations"""
#     try:
#         logger.info("Creating geographic analysis...")
#         geo_analysis = df.groupby('Country').agg({
#             'TotalAmount': 'sum',
#             'CustomerID': 'nunique',
#             'InvoiceNo': 'nunique'
#         }).reset_index()
        
#         fig = px.bar(geo_analysis.sort_values('TotalAmount', ascending=True).tail(10),
#                     x='TotalAmount',
#                     y='Country',
#                     orientation='h',
#                     title='Top 10 Countries by Sales')
        
#         return fig, geo_analysis
#     except Exception as e:
#         logger.error(f"Error in geographic analysis: {str(e)}", exc_info=True)
#         raise

# def create_customer_segmentation(analyzer):
#     """Create customer segmentation analysis"""
#     try:
#         logger.info("Starting customer segmentation analysis...")
#         customer_features = analyzer.create_customer_features()
#         clustered_customers = analyzer.perform_clustering(n_clusters=5)
#         cluster_summary = analyzer.analyze_clusters()
        
#         return clustered_customers, cluster_summary
#     except Exception as e:
#         logger.error(f"Error in customer segmentation: {str(e)}", exc_info=True)
#         raise

# def create_product_analysis(analyzer):
#     """Create product performance analysis"""
#     try:
#         logger.info("Starting product analysis...")
#         product_analysis = analyzer.analyze_product_performance()
#         top_products = product_analysis.head(10).reset_index()
        
#         fig = px.bar(top_products,
#                     x='TotalAmount',
#                     y='Description',
#                     orientation='h',
#                     title='Top 10 Products by Revenue')
        
#         return fig, product_analysis
#     except Exception as e:
#         logger.error(f"Error in product analysis: {str(e)}", exc_info=True)
#         raise

# def main():
#     try:
#         logger.info("Starting Streamlit dashboard...")
#         st.set_page_config(layout="wide")
#         st.title('Retail Analysis Dashboard (2009-2011)')
        
#         # Load data
#         try:
#             analyzer, df = load_data()
#             logger.info("Data loaded successfully")
#         except Exception as e:
#             st.error(f"Error loading data: {str(e)}")
#             st.error(f"Error type: {type(e).__name__}")
#             st.error(f"Full error details: {e.__dict__}")
#             logger.error("Failed to load data", exc_info=True)
#             return
        
#         # Sidebar filters
#         st.sidebar.title('Filters')
#         date_range = st.sidebar.date_input(
#             'Select Date Range',
#             [df['InvoiceDate'].min(), df['InvoiceDate'].max()]
#         )
        
#         selected_countries = st.sidebar.multiselect(
#             'Select Countries',
#             options=df['Country'].unique(),
#             default=df['Country'].unique()[:5]
#         )
        
#         # Filter data
#         mask = (df['InvoiceDate'].dt.date >= date_range[0]) & \
#                (df['InvoiceDate'].dt.date <= date_range[1]) & \
#                (df['Country'].isin(selected_countries))
#         filtered_df = df[mask]
        
#         # 1. Key Metrics
#         st.header('Key Metrics')
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Total Revenue", f"£{filtered_df['TotalAmount'].sum():,.2f}")
#         with col2:
#             st.metric("Total Orders", f"{filtered_df['InvoiceNo'].nunique():,}")
#         with col3:
#             st.metric("Total Customers", f"{filtered_df['CustomerID'].nunique():,}")
#         with col4:
#             avg_order = filtered_df['TotalAmount'].sum() / filtered_df['InvoiceNo'].nunique()
#             st.metric("Avg Order Value", f"£{avg_order:.2f}")
        
#         # 2. Sales Trends
#         st.header('Sales Trends')
#         monthly, quarterly, yearly = create_time_series_analysis(filtered_df)
        
#         # Sales trend visualization
#         fig = make_subplots(rows=2, cols=1, subplot_titles=('Monthly Revenue', 'Monthly Orders'))
        
#         fig.add_trace(
#             go.Scatter(x=monthly['InvoiceDate'], y=monthly['TotalAmount'], name='Revenue'),
#             row=1, col=1
#         )
        
#         fig.add_trace(
#             go.Scatter(x=monthly['InvoiceDate'], y=monthly['InvoiceNo'], name='Orders'),
#             row=2, col=1
#         )
        
#         fig.update_layout(height=600, showlegend=True)
#         st.plotly_chart(fig, use_container_width=True)
        
#         # 3. Customer Segmentation
#         st.header('Customer Segmentation')
#         clustered_customers, cluster_summary = create_customer_segmentation(analyzer)
        
#         # Display cluster summary
#         st.subheader('Cluster Characteristics')
#         st.dataframe(cluster_summary)
        
#         # Cluster visualization
#         fig = px.scatter(clustered_customers, x='TotalSpent', y='Frequency',
#                         color='Cluster', hover_data=['CustomerID'],
#                         title='Customer Segments')
#         st.plotly_chart(fig, use_container_width=True)
        
#         # 4. Geographic Analysis
#         st.header('Geographic Analysis')
#         geo_fig, geo_analysis = create_geographic_analysis(filtered_df)
#         st.plotly_chart(geo_fig, use_container_width=True)
        
#         # Display detailed country metrics
#         st.subheader('Country Details')
#         st.dataframe(geo_analysis)
        
#         # 5. Product Analysis
#         st.header('Product Analysis')
#         product_fig, product_analysis = create_product_analysis(analyzer)
#         st.plotly_chart(product_fig, use_container_width=True)
        
#         # Display top products table
#         st.subheader('Top Products Details')
#         st.dataframe(product_analysis.head(10))
        
#     except Exception as e:
#         logger.error("Critical error in main function", exc_info=True)
#         st.error(f"Error: {str(e)}")
#         st.error(f"Error type: {type(e).__name__}")
#         st.error(f"Full error details: {e.__dict__}")
#         st.warning("Please check the logs for more details and ensure S3 access is configured correctly.")

# if __name__ == '__main__':
#     main()

# src/dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the analysis directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'analysis'))
from retail_analyzer import RetailAnalyzer

def load_data():
    """Load and process data using RetailAnalyzer"""
    try:
        logger.info("Starting data loading process...")
        analyzer = RetailAnalyzer('s3://retail-analysis-data-demo/online_retail_II.xlsx')
        logger.info("RetailAnalyzer initialized, attempting to load data...")
        df = analyzer.load_and_preprocess()
        if df is None:
            logger.error("Data loading failed - DataFrame is None")
            raise Exception("Failed to load data")
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return analyzer, df
    except Exception as e:
        logger.error(f"Error in load_data: {str(e)}", exc_info=True)
        raise

def create_time_series_analysis(df):
    """Create detailed time series analysis"""
    try:
        logger.info("Starting time series analysis...")
        # Monthly trends
        monthly = df.groupby(df['InvoiceDate'].dt.to_period('M')).agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique'
        }).reset_index()
        monthly['InvoiceDate'] = monthly['InvoiceDate'].astype(str)
        
        # Quarterly trends
        df['Quarter'] = df['InvoiceDate'].dt.to_period('Q').astype(str)
        quarterly = df.groupby('Quarter').agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique'
        }).reset_index()
        
        # Year-over-Year comparison
        df['Year'] = df['InvoiceDate'].dt.year
        yearly = df.groupby('Year').agg({
            'TotalAmount': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique'
        }).reset_index()
        
        logger.info("Time series analysis completed successfully")
        return monthly, quarterly, yearly
    except Exception as e:
        logger.error(f"Error in time series analysis: {str(e)}", exc_info=True)
        raise

def display_seasonal_analysis(analyzer):
    """Display seasonal analysis insights"""
    try:
        st.header("Seasonal Analysis")
        seasonal_patterns = analyzer.analyze_seasonal_patterns()
        
        # Seasonal Sales
        st.subheader("Sales by Season")
        seasonal_sales = seasonal_patterns['seasonal_sales'].reset_index()
        fig = px.bar(seasonal_sales, 
                    x='Season', 
                    y='TotalAmount',
                    title='Sales Distribution Across Seasons')
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily Patterns
        st.subheader("Daily Sales Patterns")
        daily_patterns = seasonal_patterns['daily_patterns'].reset_index()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_patterns['DayName'] = [days[i] for i in daily_patterns['DayOfWeek']]
        fig = px.line(daily_patterns, 
                     x='DayName', 
                     y='TotalAmount',
                     title='Sales Distribution Across Days')
        st.plotly_chart(fig, use_container_width=True)
        
        # Hourly Patterns
        st.subheader("Hourly Sales Patterns")
        hourly_patterns = seasonal_patterns['hourly_patterns'].reset_index()
        fig = px.line(hourly_patterns,
                     x='Hour',
                     y='TotalAmount',
                     title='Sales Distribution Across Hours')
        st.plotly_chart(fig, use_container_width=True)
        
        # Year over Year Growth
        st.subheader("Year over Year Growth")
        yoy_growth = pd.DataFrame(seasonal_patterns['year_over_year_growth']).reset_index()
        fig = px.bar(yoy_growth,
                    x='Year',
                    y='TotalAmount',
                    title='Year over Year Growth Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying seasonal analysis: {str(e)}", exc_info=True)
        st.error("Error displaying seasonal analysis")

def display_customer_behavior(analyzer):
    """Display customer behavior insights"""
    try:
        st.header("Customer Behavior Analysis")
        purchase_patterns = analyzer.analyze_customer_behavior()
        
        # RFM Analysis
        st.subheader("Customer Segmentation (RFM Analysis)")
        rfm_analysis = purchase_patterns['rfm_analysis']
        
        # Segment Distribution
        segment_dist = rfm_analysis['Customer_Segment'].value_counts()
        fig = px.pie(values=segment_dist.values, 
                    names=segment_dist.index,
                    title='Customer Segment Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # RFM Metrics Distribution
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = px.box(rfm_analysis, y='Recency', title='Recency Distribution')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(rfm_analysis, y='Frequency', title='Frequency Distribution')
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = px.box(rfm_analysis, y='Monetary', title='Monetary Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer Lifetime Value
        st.subheader("Customer Lifetime Value Analysis")
        clv_stats = purchase_patterns['customer_lifetime_value']
        st.dataframe(pd.DataFrame(clv_stats).round(2))
        
    except Exception as e:
        logger.error(f"Error displaying customer behavior: {str(e)}", exc_info=True)
        st.error("Error displaying customer behavior analysis")

def display_clustering_comparison(analyzer):
    """Display clustering algorithm comparison"""
    try:
        st.header("Clustering Algorithm Comparison")
        clustering_results = analyzer.compare_clustering_algorithms()
        
        # K-means results
        st.subheader("K-means Clustering Performance")
        kmeans_scores = pd.DataFrame([
            {'n_clusters': res['n_clusters'], 'silhouette_score': res['silhouette_score']}
            for res in clustering_results['kmeans']
        ])
        fig = px.line(kmeans_scores, 
                     x='n_clusters',
                     y='silhouette_score',
                     title='K-means Silhouette Scores')
        st.plotly_chart(fig, use_container_width=True)
        
        # Hierarchical Clustering results
        st.subheader("Hierarchical Clustering Performance")
        hierarchical_scores = pd.DataFrame([
            {'n_clusters': res['n_clusters'], 'silhouette_score': res['silhouette_score']}
            for res in clustering_results['hierarchical']
        ])
        fig = px.line(hierarchical_scores,
                     x='n_clusters',
                     y='silhouette_score',
                     title='Hierarchical Clustering Silhouette Scores')
        st.plotly_chart(fig, use_container_width=True)
        
        # DBSCAN results
        st.subheader("DBSCAN Performance")
        if clustering_results['dbscan']:
            dbscan_scores = pd.DataFrame(clustering_results['dbscan'])
            fig = px.scatter(dbscan_scores,
                           x='eps',
                           y='silhouette_score',
                           size='min_samples',
                           title='DBSCAN Performance with Different Parameters')
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying clustering comparison: {str(e)}", exc_info=True)
        st.error("Error displaying clustering comparison")

def main():
    try:
        logger.info("Starting Streamlit dashboard...")
        st.set_page_config(layout="wide")
        st.title('Retail Analysis Dashboard (2009-2011)')
        
        # Load data and create features
        try:
            analyzer, df = load_data()
            logger.info("Data loaded successfully")
            
            # Create customer features immediately after loading
            customer_features = analyzer.create_customer_features()
            logger.info("Customer features created successfully")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            st.error(f"Full error details: {e.__dict__}")
            logger.error("Failed to load data", exc_info=True)
            return
        
        # Sidebar filters
        st.sidebar.title('Filters')
        date_range = st.sidebar.date_input(
            'Select Date Range',
            [df['InvoiceDate'].min(), df['InvoiceDate'].max()]
        )
        
        selected_countries = st.sidebar.multiselect(
            'Select Countries',
            options=df['Country'].unique(),
            default=df['Country'].unique()[:5]
        )
        
        # Filter data
        mask = (df['InvoiceDate'].dt.date >= date_range[0]) & \
               (df['InvoiceDate'].dt.date <= date_range[1]) & \
               (df['Country'].isin(selected_countries))
        filtered_df = df[mask]
        
        # 1. Key Metrics
        st.header('Key Metrics')
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Revenue", f"£{filtered_df['TotalAmount'].sum():,.2f}")
        with col2:
            st.metric("Total Orders", f"{filtered_df['InvoiceNo'].nunique():,}")
        with col3:
            st.metric("Total Customers", f"{filtered_df['CustomerID'].nunique():,}")
        with col4:
            avg_order = filtered_df['TotalAmount'].sum() / filtered_df['InvoiceNo'].nunique()
            st.metric("Avg Order Value", f"£{avg_order:.2f}")
        
        # 2. Clustering Analysis
        display_clustering_comparison(analyzer)
        
        # 3. Seasonal Analysis
        display_seasonal_analysis(analyzer)
        
        # 4. Customer Behavior Analysis
        display_customer_behavior(analyzer)
        
        # 5. Sales Trends
        st.header('Sales Trends')
        monthly, quarterly, yearly = create_time_series_analysis(filtered_df)
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Monthly Revenue', 'Monthly Orders'))
        
        fig.add_trace(
            go.Scatter(x=monthly['InvoiceDate'], y=monthly['TotalAmount'], name='Revenue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=monthly['InvoiceDate'], y=monthly['InvoiceNo'], name='Orders'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # 6. Geographic Analysis
        st.header('Geographic Analysis')
        country_analysis = analyzer.analyze_geographic_distribution()
        
        # Create choropleth map
        geo_df = country_analysis.reset_index()
        fig = px.choropleth(
            geo_df,
            locations='Country',
            locationmode='country names',
            color='TotalAmount',
            hover_data=['CustomerID', 'InvoiceNo'],
            color_continuous_scale='Viridis',
            title='Sales Distribution by Country'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed country metrics
        st.subheader('Country Details')
        st.dataframe(geo_df)
        
        # 7. Product Analysis
        st.header('Product Analysis')
        product_analysis = analyzer.analyze_product_performance()
        top_products = product_analysis.head(10).reset_index()
        
        fig = px.bar(
            top_products,
            x='TotalAmount',
            y='Description',
            orientation='h',
            title='Top 10 Products by Revenue'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display top products table
        st.subheader('Top Products Details')
        st.dataframe(top_products)
        
    except Exception as e:
        logger.error("Critical error in main function", exc_info=True)
        st.error(f"Error: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        st.error(f"Full error details: {e.__dict__}")
        st.warning("Please check the logs for more details and ensure S3 access is configured correctly.")

if __name__ == '__main__':
    main()