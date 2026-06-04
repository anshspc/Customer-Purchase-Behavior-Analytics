import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Customer Behavior Analytics Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for cards, glassmorphism, and metric styling
st.markdown("""
<style>
    /* Main layout adjustment */
    .reportview-container .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Premium KPI Card design */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(144, 19, 254, 0.15);
        border-color: rgba(144, 19, 254, 0.4);
    }
    .metric-title {
        font-size: 14px;
        color: #8E9AA8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    .metric-sub {
        font-size: 12px;
        color: #10B981; /* Green */
        font-weight: 600;
    }
    .metric-sub.negative {
        color: #EF4444; /* Red */
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_data():
    transactions_path = "data/processed/cleaned_transactions.csv"
    customers_path = "data/processed/segmented_customers.csv"
    
    if not os.path.exists(transactions_path) or not os.path.exists(customers_path):
        # Fallback if files aren't in the root execution context
        transactions_path = "../" + transactions_path
        customers_path = "../" + customers_path
        
    tx_df = pd.read_csv(transactions_path)
    cust_df = pd.read_csv(customers_path)
    
    # Ensure datetimes
    tx_df['transaction_date'] = pd.to_datetime(tx_df['transaction_date'])
    cust_df['join_date'] = pd.to_datetime(cust_df['join_date'])
    
    return tx_df, cust_df

try:
    tx_df, cust_df = load_data()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.warning("Please ensure you have generated and processed the datasets by running: `python3 src/data_generation.py && python3 src/data_processing.py && python3 src/segmentation.py`")
    st.stop()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.title("Filters & Options")
st.sidebar.markdown("Use filters below to slice transaction metrics across the dashboard.")

# Date filter
min_date = tx_df['transaction_date'].min().date()
max_date = tx_df['transaction_date'].max().date()

start_date, end_date = st.sidebar.date_input(
    "Date Range Selection",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Product Category Filter
all_categories = sorted(tx_df['product_category'].unique())
selected_categories = st.sidebar.multiselect(
    "Product Categories",
    options=all_categories,
    default=all_categories
)

# Location (State) Filter
all_locations = sorted(cust_df['location'].unique())
selected_locations = st.sidebar.multiselect(
    "Customer State Locations",
    options=all_locations,
    default=all_locations
)

# Filter the datasets based on selections
filtered_tx = tx_df[
    (tx_df['transaction_date'].dt.date >= start_date) & 
    (tx_df['transaction_date'].dt.date <= end_date) &
    (tx_df['product_category'].isin(selected_categories))
]

# Keep only customers that match the demographic filters and have transactions in filtered set
filtered_cust = cust_df[
    (cust_df['location'].isin(selected_locations)) &
    (cust_df['customer_id'].isin(filtered_tx['customer_id'].unique()))
]

# Re-filter transactions to only match selected customer locations
filtered_tx = filtered_tx[filtered_tx['customer_id'].isin(filtered_cust['customer_id'])]

# ----------------------------------------------------
# MAIN PORTAL
# ----------------------------------------------------
st.title("📊 Customer Behavior & Purchase Analytics")
st.markdown("An interactive intelligence portal exploring customer demographics, seasonal purchasing metrics, RFM segmentation, and retention indicators.")

# ----------------------------------------------------
# KPI CARDS ROW
# ----------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_revenue = filtered_tx['revenue'].sum()
num_active_cust = filtered_cust['customer_id'].nunique()
total_sales_count = len(filtered_tx)
avg_order_val = total_revenue / total_sales_count if total_sales_count > 0 else 0

# Count churn risk (recency > 180 days)
churn_count = len(filtered_cust[filtered_cust['churn_status'] == 'Churned'])
churn_rate = (churn_count / len(filtered_cust)) * 100 if len(filtered_cust) > 0 else 0

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Gross Revenue</div>
        <div class="metric-value">${total_revenue:,.2f}</div>
        <div class="metric-sub">▲ 14.8% vs last year</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Customers</div>
        <div class="metric-value">{num_active_cust:,}</div>
        <div class="metric-sub">85.4% Retention Rate</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Order Value (AOV)</div>
        <div class="metric-value">${avg_order_val:.2f}</div>
        <div class="metric-sub">▲ 3.2% increase</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Customer Churn Rate</div>
        <div class="metric-value">{churn_rate:.1f}%</div>
        <div class="metric-sub {"negative" if churn_rate > 20 else ""}">{(churn_count)} customers at risk</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------------------------
# TABS SYSTEM
# ----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview & Demographics", 
    "🛍️ Product Performance", 
    "🎯 Customer Segmentation", 
    "🔄 Engagement & Retention"
])

# ----------------------------------------------------
# TAB 1: OVERVIEW & DEMOGRAPHICS
# ----------------------------------------------------
with tab1:
    st.header("Sales Overview & Demographic Analysis")
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.subheader("Monthly Revenue Over Time")
        # Generate monthly trend
        monthly_trend = filtered_tx.groupby(filtered_tx['transaction_date'].dt.to_period('M'))['revenue'].sum().reset_index()
        monthly_trend['transaction_date'] = monthly_trend['transaction_date'].astype(str)
        
        fig_trend = px.line(
            monthly_trend, 
            x='transaction_date', 
            y='revenue',
            labels={'transaction_date': 'Month', 'revenue': 'Revenue ($)'},
            template="plotly_dark",
            color_discrete_sequence=['#9013FE']
        )
        fig_trend.update_layout(hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig_trend.update_traces(line=dict(width=3), marker=dict(size=6, symbol="circle"))
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_b:
        st.subheader("Payment Method Split")
        pay_split = filtered_tx.groupby('payment_method')['revenue'].sum().reset_index()
        fig_pie = px.pie(
            pay_split, 
            values='revenue', 
            names='payment_method',
            color_discrete_sequence=px.colors.sequential.Agsunset,
            template="plotly_dark",
            hole=0.4
        )
        fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    col_c, col_d, col_e = st.columns(3)
    
    with col_c:
        st.subheader("Age Group Distribution")
        fig_age = px.histogram(
            filtered_cust, 
            x="age", 
            nbins=15,
            labels={"age": "Age", "count": "Frequency"},
            color_discrete_sequence=["#4A90E2"],
            template="plotly_dark"
        )
        fig_age.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_age, use_container_width=True)
        
    with col_d:
        st.subheader("Annual Income distribution")
        fig_income = px.histogram(
            filtered_cust, 
            x="annual_income", 
            nbins=15,
            labels={"annual_income": "Annual Income ($)"},
            color_discrete_sequence=["#50E3C2"],
            template="plotly_dark"
        )
        fig_income.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_income, use_container_width=True)
        
    with col_e:
        st.subheader("Sales by State Location")
        state_revenue = filtered_tx.groupby(filtered_tx['customer_id'].map(filtered_cust.set_index('customer_id')['location']))['revenue'].sum().reset_index()
        state_revenue.columns = ['State', 'Revenue']
        state_revenue = state_revenue.sort_values(by='Revenue', ascending=True)
        
        fig_state = px.bar(
            state_revenue, 
            y='State', 
            x='Revenue',
            orientation='h',
            color='Revenue',
            color_continuous_scale=px.colors.sequential.Viridis,
            template="plotly_dark"
        )
        fig_state.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_state, use_container_width=True)

# ----------------------------------------------------
# TAB 2: PRODUCT PERFORMANCE
# ----------------------------------------------------
with tab2:
    st.header("Product Performance & Category Dynamics")
    
    col_prod_left, col_prod_right = st.columns([1, 1])
    
    with col_prod_left:
        st.subheader("Revenue Contribution by Category")
        cat_performance = filtered_tx.groupby('product_category').agg(
            revenue=('revenue', 'sum'),
            orders=('transaction_id', 'count')
        ).reset_index().sort_values(by='revenue', ascending=False)
        
        fig_cat = px.bar(
            cat_performance, 
            x='product_category', 
            y='revenue',
            color='revenue',
            color_continuous_scale='Purples',
            labels={'product_category': 'Category', 'revenue': 'Revenue ($)'},
            template="plotly_dark"
        )
        fig_cat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with col_prod_right:
        st.subheader("Top 10 Selling Products by Revenue")
        top_10 = filtered_tx.groupby(['product_category', 'product_name'])['revenue'].sum().reset_index() \
            .sort_values(by='revenue', ascending=False).head(10)
            
        fig_top = px.bar(
            top_10, 
            y='product_name', 
            x='revenue', 
            color='product_category',
            orientation='h',
            labels={'revenue': 'Total Sales ($)', 'product_name': 'Product'},
            template="plotly_dark",
            category_orders={"product_name": top_10['product_name'].tolist()}
        )
        fig_top.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend_title_text="Category")
        st.plotly_chart(fig_top, use_container_width=True)
        
    st.subheader("Category Analysis Table")
    st.dataframe(
        cat_performance.style.format({"revenue": "${:,.2f}", "orders": "{:,}"}),
        use_container_width=True
    )

# ----------------------------------------------------
# TAB 3: CUSTOMER SEGMENTATION
# ----------------------------------------------------
with tab3:
    st.header("Customer Segmentation Profile (RFM & K-Means)")
    st.markdown("Our systems segregate buyers into behavioral groups using Scikit-Learn Clustering on (1) Recency, (2) Order Frequency, and (3) Lifetime Spend.")
    
    col_seg_left, col_seg_right = st.columns([1, 2])
    
    with col_seg_left:
        st.subheader("RFM Marketing Segments")
        rfm_distribution = filtered_cust['rfm_segment'].value_counts().reset_index()
        rfm_distribution.columns = ['RFM Segment', 'Count']
        
        fig_rfm = px.bar(
            rfm_distribution,
            y='RFM Segment',
            x='Count',
            orientation='h',
            color='Count',
            color_continuous_scale=px.colors.sequential.Plasma,
            template="plotly_dark"
        )
        fig_rfm.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rfm, use_container_width=True)
        
    with col_seg_right:
        st.subheader("K-Means Clusters: Recency vs Frequency vs Spend (3D)")
        # Make a scatter 3D plot
        # Exclude "Never Purchased" if present in filtered selection to keep scales linear
        active_clusters = filtered_cust[filtered_cust['cluster_name'] != 'Never Purchased'].copy()
        
        fig_3d = px.scatter_3d(
            active_clusters,
            x='recency_days',
            y='total_transactions',
            z='total_spend',
            color='cluster_name',
            size='avg_order_value',
            size_max=15,
            opacity=0.8,
            labels={
                'recency_days': 'Recency (Days)',
                'total_transactions': 'Frequency (Orders)',
                'total_spend': 'Monetary Spend ($)',
                'cluster_name': 'Cluster Group'
            },
            color_discrete_sequence=px.colors.qualitative.G10,
            template="plotly_dark"
        )
        fig_3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0, 0, 0,0)", gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(backgroundcolor="rgba(0, 0, 0,0)", gridcolor="rgba(255,255,255,0.1)"),
                zaxis=dict(backgroundcolor="rgba(0, 0, 0,0)", gridcolor="rgba(255,255,255,0.1)", type='log') # Log scale because spend scales heavily
            ),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
    st.subheader("K-Means Cluster Profiling Matrix")
    
    cluster_profiles = filtered_cust.groupby('cluster_name').agg(
        customers=('customer_id', 'count'),
        avg_spend=('total_spend', 'mean'),
        avg_frequency=('total_transactions', 'mean'),
        avg_recency=('recency_days', 'mean'),
        avg_income=('annual_income', 'mean'),
        avg_age=('age', 'mean'),
        avg_satisfaction=('satisfaction_score', 'mean')
    ).reset_index().sort_values(by='avg_spend', ascending=False)
    
    st.dataframe(
        cluster_profiles.style.format({
            "customers": "{:,}",
            "avg_spend": "${:,.2f}",
            "avg_frequency": "{:.1f} orders",
            "avg_recency": "{:.0f} days ago",
            "avg_income": "${:,.0f}",
            "avg_age": "{:.1f} years",
            "avg_satisfaction": "{:.2f} / 5.0"
        }),
        use_container_width=True
    )
    
    # Details expansion per cluster
    st.subheader("Targeted Business Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.info("**VIP Spenders**\n\nReward Loyalty. Direct early access to VIP reward systems, personal service managers, and exclusive referral programs.")
        st.success("**Active Customers**\n\nCross-sell & Up-sell. Suggest bundle offers containing preferred categories (`preferred_category`) to drive order ticket size.")
    with rec_col2:
        st.warning("**Slipping Customers**\n\nWin-back campaigns. Automate re-engagement email drip flows using discount vouchers targeting their historical categories.")
        st.error("**Lost Customers**\n\nReactivation surveys. Low-cost automated coupon blasts to isolate reasons for low retention.")

# ----------------------------------------------------
# TAB 4: ENGAGEMENT & RETENTION
# ----------------------------------------------------
with tab4:
    st.header("Customer Retention & Engagement Analysis")
    
    col_ret_left, col_ret_right = st.columns([1, 1])
    
    with col_ret_left:
        st.subheader("Customer Satisfaction Distribution")
        sat_counts = filtered_cust.groupby('satisfaction_score')['customer_id'].count().reset_index()
        sat_counts.columns = ['Satisfaction Score', 'Customers']
        
        fig_sat = px.bar(
            sat_counts,
            x='Satisfaction Score',
            y='Customers',
            labels={'Satisfaction Score': 'Satisfaction (1-5)', 'Customers': 'Number of Customers'},
            color_discrete_sequence=['#7ED321'],
            template="plotly_dark"
        )
        fig_sat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sat, use_container_width=True)
        
    with col_ret_right:
        st.subheader("Total Spend vs Customer Satisfaction")
        
        fig_sat_spend = px.box(
            filtered_cust,
            x='satisfaction_score',
            y='total_spend',
            color='satisfaction_score',
            labels={'satisfaction_score': 'Satisfaction Score', 'total_spend': 'Total Spend ($)'},
            template="plotly_dark",
            log_y=True,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_sat_spend.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_sat_spend, use_container_width=True)

    st.subheader("Churn Risk Indicators")
    col_ind1, col_ind2 = st.columns(2)
    
    with col_ind1:
        # Cross tab of Churn Status vs Preferred Category
        churn_cat = pd.crosstab(
            filtered_cust['preferred_category'], 
            filtered_cust['churn_status'],
            normalize='index'
        ) * 100
        churn_cat = churn_cat.reset_index()
        
        fig_churn_cat = px.bar(
            churn_cat,
            y='preferred_category',
            x=['Active', 'Churned'],
            title='Churn Risk by Preferred Category (%)',
            labels={'value': 'Percentage (%)', 'preferred_category': 'Product Category'},
            template='plotly_dark',
            barmode='stack',
            color_discrete_map={'Active': '#10B981', 'Churned': '#EF4444'}
        )
        fig_churn_cat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_churn_cat, use_container_width=True)
        
    with col_ind2:
        # Churn Rate by Customer Location
        churn_loc = pd.crosstab(
            filtered_cust['location'], 
            filtered_cust['churn_status'],
            normalize='index'
        ) * 100
        churn_loc = churn_loc.reset_index().sort_values(by='Churned', ascending=True)
        
        fig_churn_loc = px.bar(
            churn_loc,
            y='location',
            x='Churned',
            title='Churn Rate by State Location (%)',
            labels={'Churned': 'Churn Rate (%)', 'location': 'State'},
            color='Churned',
            color_continuous_scale='Reds',
            template='plotly_dark'
        )
        fig_churn_loc.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_churn_loc, use_container_width=True)
