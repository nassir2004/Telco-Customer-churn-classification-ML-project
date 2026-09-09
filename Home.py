import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    load_data, 
    apply_custom_css, 
    render_hero_banner, 
    render_metric_card, 
    render_takeaway,
    format_plotly_chart,
    CHURN_COLOR_MAP
)

# Set page config
st.set_page_config(
    page_title="Telco Churn Hub | Overview", 
    page_icon="🏠", 
    layout="wide"
)

# Apply sleek styling
apply_custom_css()

# Load cleaned dataset
df = load_data()

# Hero Header
render_hero_banner(
    title="Telco Customer Churn Analytics & Prediction",
    subtitle="Executive overview of customer retention performance, revenue metrics, and machine learning insight."
)

# Key Executive Metrics
total_customers = len(df)
churn_count = (df['Churn'] == 'Yes').sum()
churn_rate = (churn_count / total_customers) * 100
avg_monthly = df['MonthlyCharges'].mean()
avg_tenure = df['tenure'].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card("Total Customers", f"{total_customers:,}", "Active Customer Base")

with col2:
    render_metric_card("Overall Churn Rate", f"{churn_rate:.2f}%", f"{churn_count:,} Customers Lost", is_negative=True)

with col3:
    render_metric_card("Avg Monthly Charges", f"${avg_monthly:.2f}", "Per Customer / Month")

with col4:
    render_metric_card("Avg Customer Tenure", f"{avg_tenure:.1f} Mo", f"~{avg_tenure/12:.1f} Years Average")

st.markdown("<br>", unsafe_allow_html=True)

# Main Dashboard Grid
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("📌 Overall Churn Breakdown")
    churn_df = df['Churn'].value_counts().reset_index()
    churn_df.columns = ['Churn', 'Count']
    churn_df['Percentage'] = (churn_df['Count'] / total_customers * 100).round(1)
    
    fig_pie = px.pie(
        churn_df, 
        names='Churn', 
        values='Count',
        color='Churn',
        color_discrete_map=CHURN_COLOR_MAP,
        hole=0.55,
        hover_data=['Percentage']
    )
    fig_pie.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    fig_pie = format_plotly_chart(fig_pie, title="Customer Retention vs. Churn Ratio", height=350)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    render_takeaway("26.5% of overall customers have churned. Retaining these customers represents a substantial revenue recovery opportunity.")

with c2:
    st.subheader("📌 Churn Rate by Contract Type")
    contract_churn = df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
    contract_totals = df.groupby('Contract').size().reset_index(name='Total')
    contract_merged = pd.merge(contract_churn, contract_totals, on='Contract')
    contract_merged['Percentage'] = (contract_merged['Count'] / contract_merged['Total'] * 100).round(1)
    
    fig_bar = px.bar(
        contract_merged,
        x='Contract',
        y='Count',
        color='Churn',
        barmode='stack',
        color_discrete_map=CHURN_COLOR_MAP,
        text='Percentage',
        hover_data=['Percentage', 'Total']
    )
    fig_bar.update_traces(texttemplate='%{text}%', textposition='inside')
    fig_bar = format_plotly_chart(fig_bar, title="Contract Commitment vs Churn Volume", height=350)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    render_takeaway("Month-to-month contracts experience dramatically higher churn compared to 1-year and 2-year commitments.")

st.markdown("---")

# Service & Tenure Overview
c3, c4 = st.columns([1, 1])

with c3:
    st.subheader("📌 Internet Service Impact")
    net_churn = df.groupby(['InternetService', 'Churn']).size().reset_index(name='Count')
    fig_net = px.bar(
        net_churn,
        x='InternetService',
        y='Count',
        color='Churn',
        barmode='group',
        color_discrete_map=CHURN_COLOR_MAP,
        title="Churn Volume by Internet Service Type"
    )
    fig_net = format_plotly_chart(fig_net, height=350)
    st.plotly_chart(fig_net, use_container_width=True)
    render_takeaway("Fiber Optic users exhibit the highest churn volume due to higher monthly costs and service expectations.")

with c4:
    st.subheader("📌 Tenure vs. Monthly Charges Distribution")
    fig_scatter = px.scatter(
        df,
        x='tenure',
        y='MonthlyCharges',
        color='Churn',
        color_discrete_map=CHURN_COLOR_MAP,
        opacity=0.5,
        labels={'tenure': 'Tenure (Months)', 'MonthlyCharges': 'Monthly Charges ($)'}
    )
    fig_scatter = format_plotly_chart(fig_scatter, title="Tenure vs Monthly Charges Overview", height=350)
    st.plotly_chart(fig_scatter, use_container_width=True)
    render_takeaway("High monthly charges in the early tenure phase (0 - 12 months) represent the highest risk customer segment.")

st.markdown("---")

# Data Sample Explorer
st.subheader("🔍 Interactive Dataset Preview")
with st.expander("Click to View Dataset Sample & Search Records", expanded=False):
    st.markdown("Filter and inspect the underlying clean dataset used for model training and analysis.")
    search_col1, search_col2 = st.columns([1, 2])
    with search_col1:
        contract_filter = st.multiselect("Filter by Contract Type", options=df['Contract'].unique(), default=df['Contract'].unique())
    with search_col2:
        internet_filter = st.multiselect("Filter by Internet Service", options=df['InternetService'].unique(), default=df['InternetService'].unique())
        
    filtered_df = df[(df['Contract'].isin(contract_filter)) & (df['InternetService'].isin(internet_filter))]
    st.dataframe(filtered_df.head(100), use_container_width=True)
    st.caption(f"Showing first 100 rows out of {len(filtered_df):,} filtered records.")

# Quick Navigation Section
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🚀 Quick Application Navigation")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.info("📊 **Exploratory Churn Analysis**\n\nDive deep into customer demographics, internet services, contract types, and payment methods to discover key churn drivers.")

with nav_col2:
    st.success("🔮 **AI Churn Risk Predictor**\n\nInput custom customer profiles to evaluate churn probability in real time using our XGBoost machine learning model.")