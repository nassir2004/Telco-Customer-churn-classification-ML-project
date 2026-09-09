import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    load_data, 
    apply_custom_css, 
    render_hero_banner, 
    render_takeaway,
    format_plotly_chart,
    CHURN_COLOR_MAP
)

# Page configuration
st.set_page_config(
    page_title="Telco Churn Analysis | Deep Dive", 
    page_icon="📊", 
    layout="wide"
)

# Apply sleek styling
apply_custom_css()

# Load dataset
df = load_data()

# Hero Header
render_hero_banner(
    title="Target Churn Analysis & Key Drivers",
    subtitle="In-depth exploratory analysis focusing on customer churn across demographics, services, contract terms, and financial metrics."
)

# Define helper function for categorical churn percentage chart
def create_churn_bar_chart(df, cat_col, title):
    grouped = df.groupby([cat_col, 'Churn']).size().reset_index(name='Count')
    totals = df.groupby(cat_col).size().reset_index(name='Total')
    merged = pd.merge(grouped, totals, on=cat_col)
    merged['Percentage'] = (merged['Count'] / merged['Total'] * 100).round(1)
    
    fig = px.bar(
        merged,
        x=cat_col,
        y='Count',
        color='Churn',
        barmode='group',
        color_discrete_map=CHURN_COLOR_MAP,
        text='Percentage',
        hover_data=['Total', 'Percentage']
    )
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig = format_plotly_chart(fig, title=title, height=380)
    return fig

# Tabbed Interface
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Demographics", 
    "🌐 Services & Support", 
    "📜 Contracts & Billing", 
    "💰 Tenure & Financials"
])

# =========================================================
# TAB 1: DEMOGRAPHICS
# =========================================================
with tab1:
    st.subheader("Demographic Profile vs. Churn")
    st.markdown("Explore how customer personal background influences retention rates.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_gender = create_churn_bar_chart(df, 'gender', "Churn by Gender")
        st.plotly_chart(fig_gender, use_container_width=True)
        render_takeaway("Churn rates are almost identical between Male and Female customers (~26.9% vs ~26.2%), indicating gender is not a primary driver of churn.")
        
        fig_partner = create_churn_bar_chart(df, 'Partner', "Churn by Partner Status")
        st.plotly_chart(fig_partner, use_container_width=True)
        render_takeaway("Single customers without a partner experience significantly higher churn (32.9%) compared to customers with partners (19.7%).")
        
    with col2:
        # Senior Citizen mapping
        df_senior = df.copy()
        df_senior['SeniorCitizen_Label'] = df_senior['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior Citizen'})
        fig_senior = create_churn_bar_chart(df_senior, 'SeniorCitizen_Label', "Churn by Senior Citizen Status")
        st.plotly_chart(fig_senior, use_container_width=True)
        render_takeaway("Senior Citizens suffer a strikingly high churn rate of 41.7%, compared to only 23.6% for non-seniors.")
        
        fig_dep = create_churn_bar_chart(df, 'Dependents', "Churn by Dependents Status")
        st.plotly_chart(fig_dep, use_container_width=True)
        render_takeaway("Customers without dependents are much more likely to churn (31.3%) than those with families/dependents (15.5%).")

# =========================================================
# TAB 2: SERVICES & SUPPORT
# =========================================================
with tab2:
    st.subheader("Subscribed Telco & Internet Services")
    st.markdown("Examine how specific product add-ons and support services affect customer loyalty.")
    
    c1, c2 = st.columns(2)
    
    with c1:
        fig_net = create_churn_bar_chart(df, 'InternetService', "Churn by Internet Service Type")
        st.plotly_chart(fig_net, use_container_width=True)
        render_takeaway("Fiber Optic internet subscribers have an alarming 41.9% churn rate, compared to 19.0% for DSL and 7.4% for customers with no internet service.")
        
        fig_tech = create_churn_bar_chart(df, 'TechSupport', "Churn by Tech Support Availability")
        st.plotly_chart(fig_tech, use_container_width=True)
        render_takeaway("Customers without Tech Support churn at 41.6%, while those with Tech Support churn at only 15.2%. Support services strongly improve retention!")
        
    with c2:
        fig_sec = create_churn_bar_chart(df, 'OnlineSecurity', "Churn by Online Security Service")
        st.plotly_chart(fig_sec, use_container_width=True)
        render_takeaway("Subscribers without Online Security experience 41.8% churn versus 14.6% for secured accounts.")
        
        # Total Services Count Analysis
        fig_tot_serv = create_churn_bar_chart(df, 'Total_Services', "Churn by Total Services Subscribed")
        st.plotly_chart(fig_tot_serv, use_container_width=True)
        render_takeaway("Customers with only 1 or 2 active services have the highest churn rates. As subscribers adopt 3+ services, customer stickiness increases significantly.")

# =========================================================
# TAB 3: CONTRACTS & BILLING
# =========================================================
with tab3:
    st.subheader("Contract Duration & Payment Preferences")
    st.markdown("Analyze how financial contract structures and payment methods drive customer departure.")
    
    c3, c4 = st.columns(2)
    
    with c3:
        fig_contract = create_churn_bar_chart(df, 'Contract', "Churn Rate across Contract Terms")
        st.plotly_chart(fig_contract, use_container_width=True)
        render_takeaway("Month-to-Month contract holders have a massive 42.7% churn rate! 1-Year (11.3%) and 2-Year (2.8%) contracts provide extreme retention stability.")
        
        fig_paperless = create_churn_bar_chart(df, 'PaperlessBilling', "Churn by Paperless Billing")
        st.plotly_chart(fig_paperless, use_container_width=True)
        render_takeaway("Paperless billing users experience higher churn (33.6%), likely correlated with online payment friction and digital notification sensitivity.")
        
    with c4:
        fig_payment = create_churn_bar_chart(df, 'PaymentMethod', "Churn by Payment Method")
        st.plotly_chart(fig_payment, use_container_width=True)
        render_takeaway("Electronic Check users have an exceptionally high churn rate of 45.3%. Automated bank transfers and credit cards average under 17% churn.")
        
        # Auto Payment flag
        df_auto = df.copy()
        df_auto['AutoPayment_Label'] = df_auto['Is_Auto_Payment'].map({1: 'Automated Payment', 0: 'Manual Payment'})
        fig_auto = create_churn_bar_chart(df_auto, 'AutoPayment_Label', "Churn by Payment Automation")
        st.plotly_chart(fig_auto, use_container_width=True)
        render_takeaway("Customers on Automated Payment plans churn half as often (15.9%) as customers paying manually (34.5%).")

# =========================================================
# TAB 4: TENURE & FINANCIALS
# =========================================================
with tab4:
    st.subheader("Tenure & Pricing Metrics")
    st.markdown("Evaluate continuous numerical variables (Tenure, Monthly Charges, Total Charges) against customer churn.")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        # Tenure Boxplot
        fig_box_tenure = px.box(
            df,
            x='Churn',
            y='tenure',
            color='Churn',
            color_discrete_map=CHURN_COLOR_MAP,
            points='outliers',
            labels={'tenure': 'Tenure (Months)', 'Churn': 'Customer Status'}
        )
        fig_box_tenure = format_plotly_chart(fig_box_tenure, title="Tenure Distribution by Churn Status", height=380)
        st.plotly_chart(fig_box_tenure, use_container_width=True)
        render_takeaway("Churned customers have a median tenure of only 10 months, whereas retained customers have a median tenure of 38 months.")
        
        # Monthly Charges Boxplot
        fig_box_monthly = px.box(
            df,
            x='Churn',
            y='MonthlyCharges',
            color='Churn',
            color_discrete_map=CHURN_COLOR_MAP,
            points='outliers',
            labels={'MonthlyCharges': 'Monthly Charges ($)', 'Churn': 'Customer Status'}
        )
        fig_box_monthly = format_plotly_chart(fig_box_monthly, title="Monthly Charges Comparison ($)", height=380)
        st.plotly_chart(fig_box_monthly, use_container_width=True)
        render_takeaway("Churned customers pay significantly higher monthly fees (median ~$80/mo) compared to retained customers (median ~$64/mo).")

    with col_f2:
        # Tenure Histogram / KDE plot
        fig_hist_tenure = px.histogram(
            df,
            x='tenure',
            color='Churn',
            barmode='overlay',
            color_discrete_map=CHURN_COLOR_MAP,
            opacity=0.6,
            nbins=30,
            labels={'tenure': 'Tenure (Months)'}
        )
        fig_hist_tenure = format_plotly_chart(fig_hist_tenure, title="Customer Tenure Density Distribution", height=380)
        st.plotly_chart(fig_hist_tenure, use_container_width=True)
        render_takeaway("The first 1 to 6 months represent the 'Danger Zone' where churn spikes highest. Survival probability increases substantially after Year 1.")
        
        # Total Charges Boxplot
        fig_box_total = px.box(
            df,
            x='Churn',
            y='TotalCharges',
            color='Churn',
            color_discrete_map=CHURN_COLOR_MAP,
            points='outliers',
            labels={'TotalCharges': 'Total Charges ($)', 'Churn': 'Customer Status'}
        )
        fig_box_total = format_plotly_chart(fig_box_total, title="Total Charges Distribution ($)", height=380)
        st.plotly_chart(fig_box_total, use_container_width=True)
        render_takeaway("Because churned customers leave early, their cumulative Total Charges remain far lower despite paying higher monthly rates.")
