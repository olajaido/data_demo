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

# Add the analysis directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'analysis'))
from retail_analyzer import RetailAnalyzer

def load_data():
    """Load and process data using RetailAnalyzer"""
    analyzer = RetailAnalyzer('s3://retail-analysis-data-demo/online_retail_II.xlsx')
    df = analyzer.load_and_preprocess()
    return analyzer, df

def create_time_series_analysis(df):
    """Create detailed time series analysis"""
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
    
    return monthly, quarterly, yearly

def main():
    st.set_page_config(layout="wide")
    st.title('Retail Analysis Dashboard (2009-2011)')
    
    try:
        # Load data
        analyzer, df = load_data()
        
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
        
        # Filter data based on selections
        mask = (df['InvoiceDate'].dt.date >= date_range[0]) & \
               (df['InvoiceDate'].dt.date <= date_range[1]) & \
               (df['Country'].isin(selected_countries))
        filtered_df = df[mask]
        
        # 1. SALES TRENDS SECTION
        st.header('1. Sales Trends Analysis')
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Revenue", f"£{filtered_df['TotalAmount'].sum():,.2f}")
        with col2:
            st.metric("Total Orders", f"{filtered_df['InvoiceNo'].nunique():,}")
        with col3:
            st.metric("Total Customers", f"{filtered_df['CustomerID'].nunique():,}")
        with col4:
            avg_order_value = filtered_df['TotalAmount'].sum() / filtered_df['InvoiceNo'].nunique()
            st.metric("Avg Order Value", f"£{avg_order_value:.2f}")
        
        # Time series analysis
        monthly, quarterly, yearly = create_time_series_analysis(filtered_df)
        
        # Multiple time series views
        time_period = st.selectbox('Select Time Period', ['Monthly', 'Quarterly', 'Yearly'])
        if time_period == 'Monthly':
            data = monthly
            x_col = 'InvoiceDate'
        elif time_period == 'Quarterly':
            data = quarterly
            x_col = 'Quarter'
        else:
            data = yearly
            x_col = 'Year'
            
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Revenue Trend', 'Customer and Order Trends'))
        
        # Revenue trend
        fig.add_trace(
            go.Scatter(x=data[x_col], y=data['TotalAmount'], name='Revenue'),
            row=1, col=1
        )
        
        # Customers and Orders
        fig.add_trace(
            go.Scatter(x=data[x_col], y=data['CustomerID'], name='Unique Customers'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=data[x_col], y=data['InvoiceNo'], name='Number of Orders'),
            row=2, col=1
        )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. CUSTOMER SEGMENTATION SECTION
        st.header('2. Customer Segmentation Analysis')
        
        # Create and analyze customer segments
        customer_features = analyzer.create_customer_features()
        clustered_customers = analyzer.perform_clustering(n_clusters=5)
        cluster_summary = analyzer.analyze_clusters()
        
        # Display cluster characteristics
        st.subheader('Customer Segment Characteristics')
        cluster_summary_formatted = cluster_summary.style.format({
            'Frequency': '{:.0f}',
            'TotalSpent': '£{:,.2f}',
            'AvgTransactionValue': '£{:.2f}',
            'CustomerLifespan': '{:.0f} days',
            'AvgPurchaseFrequency': '{:.2f}',
            'CustomerID': '{:.0f}'
        })
        st.dataframe(cluster_summary_formatted)
        
        # Customer behavior patterns
        st.subheader('Customer Behavior Patterns')
        behavior_cols = ['Frequency', 'TotalSpent', 'AvgTransactionValue']
        for col in behavior_cols:
            fig = px.box(clustered_customers, x='Cluster', y=col, 
                        title=f'Distribution of {col} by Cluster')
            st.plotly_chart(fig, use_container_width=True)
        
        # 3. PRODUCT PERFORMANCE SECTION
        st.header('3. Product Performance Analysis')
        
        # Top products analysis
        product_analysis = analyzer.analyze_product_performance()
        top_products = product_analysis.head(10).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Top 10 Products by Revenue')
            fig = px.bar(top_products, 
                        x='TotalAmount',
                        y='Description',
                        orientation='h',
                        title='Top 10 Products by Revenue')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader('Top 10 Products by Quantity Sold')
            top_by_quantity = product_analysis.sort_values('Quantity', ascending=False).head(10).reset_index()
            fig = px.bar(top_by_quantity,
                        x='Quantity',
                        y='Description',
                        orientation='h',
                        title='Top 10 Products by Quantity')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
        # Product performance matrix
        st.subheader('Product Performance Matrix')
        product_matrix = product_analysis.copy()
        product_matrix['AvgPrice'] = product_matrix['TotalAmount'] / product_matrix['Quantity']
        
        fig = px.scatter(product_matrix.reset_index(),
                        x='Quantity',
                        y='TotalAmount',
                        size='AvgPrice',
                        hover_data=['Description'],
                        title='Product Performance Matrix')
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. GEOGRAPHIC INSIGHTS SECTION
        st.header('4. Geographic Insights')
        
        # Geographic distribution
        geo_analysis = analyzer.analyze_geographic_distribution()
        geo_df = geo_analysis.reset_index()
        
        # Sales by country
        fig = px.bar(geo_df,
                    x='Country',
                    y='TotalAmount',
                    title='Sales by Country')
        st.plotly_chart(fig, use_container_width=True)
        
        # Country growth analysis
        country_growth = df.pivot_table(
            index='Country',
            columns=df['InvoiceDate'].dt.year,
            values='TotalAmount',
            aggfunc='sum'
        ).fillna(0)
        
        country_growth['Growth'] = (country_growth[2011] - country_growth[2010]) / country_growth[2010] * 100
        
        st.subheader('Country Growth Analysis')
        growth_df = country_growth.sort_values('Growth', ascending=False).reset_index()
        fig = px.bar(growth_df,
                    x='Country',
                    y='Growth',
                    title='Year-over-Year Growth by Country (%)')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.warning("Please make sure your data is uploaded to S3 and the path is correct.")

if __name__ == '__main__':
    main()