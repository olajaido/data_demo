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

def create_geographic_analysis(df):
    """Create geographic analysis visualizations"""
    try:
        logger.info("Creating geographic analysis...")
        geo_analysis = df.groupby('Country').agg({
            'TotalAmount': 'sum',
            'CustomerID': 'nunique',
            'InvoiceNo': 'nunique'
        }).reset_index()
        
        fig = px.bar(geo_analysis.sort_values('TotalAmount', ascending=True).tail(10),
                    x='TotalAmount',
                    y='Country',
                    orientation='h',
                    title='Top 10 Countries by Sales')
        
        return fig, geo_analysis
    except Exception as e:
        logger.error(f"Error in geographic analysis: {str(e)}", exc_info=True)
        raise

def create_customer_segmentation(analyzer):
    """Create customer segmentation analysis"""
    try:
        logger.info("Starting customer segmentation analysis...")
        customer_features = analyzer.create_customer_features()
        clustered_customers = analyzer.perform_clustering(n_clusters=5)
        cluster_summary = analyzer.analyze_clusters()
        
        return clustered_customers, cluster_summary
    except Exception as e:
        logger.error(f"Error in customer segmentation: {str(e)}", exc_info=True)
        raise

def create_product_analysis(analyzer):
    """Create product performance analysis"""
    try:
        logger.info("Starting product analysis...")
        product_analysis = analyzer.analyze_product_performance()
        top_products = product_analysis.head(10).reset_index()
        
        fig = px.bar(top_products,
                    x='TotalAmount',
                    y='Description',
                    orientation='h',
                    title='Top 10 Products by Revenue')
        
        return fig, product_analysis
    except Exception as e:
        logger.error(f"Error in product analysis: {str(e)}", exc_info=True)
        raise

def main():
    try:
        logger.info("Starting Streamlit dashboard...")
        st.set_page_config(layout="wide")
        st.title('Retail Analysis Dashboard (2009-2011)')
        
        # Load data
        try:
            analyzer, df = load_data()
            logger.info("Data loaded successfully")
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
        
        # 2. Sales Trends
        st.header('Sales Trends')
        monthly, quarterly, yearly = create_time_series_analysis(filtered_df)
        
        # Sales trend visualization
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Monthly Revenue', 'Monthly Orders'))
        
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
        
        # 3. Customer Segmentation
        st.header('Customer Segmentation')
        clustered_customers, cluster_summary = create_customer_segmentation(analyzer)
        
        # Display cluster summary
        st.subheader('Cluster Characteristics')
        st.dataframe(cluster_summary)
        
        # Cluster visualization
        fig = px.scatter(clustered_customers, x='TotalSpent', y='Frequency',
                        color='Cluster', hover_data=['CustomerID'],
                        title='Customer Segments')
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. Geographic Analysis
        st.header('Geographic Analysis')
        geo_fig, geo_analysis = create_geographic_analysis(filtered_df)
        st.plotly_chart(geo_fig, use_container_width=True)
        
        # Display detailed country metrics
        st.subheader('Country Details')
        st.dataframe(geo_analysis)
        
        # 5. Product Analysis
        st.header('Product Analysis')
        product_fig, product_analysis = create_product_analysis(analyzer)
        st.plotly_chart(product_fig, use_container_width=True)
        
        # Display top products table
        st.subheader('Top Products Details')
        st.dataframe(product_analysis.head(10))
        
    except Exception as e:
        logger.error("Critical error in main function", exc_info=True)
        st.error(f"Error: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        st.error(f"Full error details: {e.__dict__}")
        st.warning("Please check the logs for more details and ensure S3 access is configured correctly.")

if __name__ == '__main__':
    main()