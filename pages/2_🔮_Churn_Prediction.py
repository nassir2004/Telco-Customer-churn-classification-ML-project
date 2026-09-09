import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost
from utils import (
    load_data, 
    load_pipeline, 
    apply_custom_css, 
    render_hero_banner,
    format_plotly_chart
)

# Page configuration
st.set_page_config(
    page_title="Telco Churn Predictor | AI Model", 
    page_icon="🔮", 
    layout="wide"
)

# Apply custom styles
apply_custom_css()

# Load clean dataset and pre-trained XGBoost pipeline
data = load_data()
pipeline = load_pipeline()

# Hero Banner
render_hero_banner(
    title="AI Customer Churn Predictor",
    subtitle="Assess individual customer churn risk probabilities using our pre-trained XGBoost pipeline model."
)

# =========================================================
# SIDEBAR NUMERICAL INPUTS (AS REQUESTED)
# =========================================================
st.sidebar.header("⚙️ Numerical Features")
st.sidebar.markdown("Adjust key billing and tenure metrics:")

# Tenure (Months up to 240)
tenure = st.sidebar.number_input(
    "Tenure (Months)", 
    min_value=0, 
    max_value=240, 
    value=12, 
    step=1,
    help="Duration customer has stayed with the company in months."
)

# Monthly Charges (Up to $300)
monthly_charges = st.sidebar.number_input(
    "Monthly Charges ($)", 
    min_value=0.0, 
    max_value=300.0,
    value=65.0, 
    step=1.0,
    format="%.2f",
    help="Current monthly billing charge in USD."
)

# Total Charges (Up to $25,000)
total_charges = st.sidebar.number_input(
    "Total Charges ($)", 
    min_value=0.0, 
    max_value=25000.0,
    value=500.0, 
    step=10.0,
    format="%.2f",
    help="Cumulative overall charges billed to customer."
)

# Total Services Count Slider (0 to 9)
total_services = st.sidebar.slider(
    "Total Services Count", 
    min_value=0, 
    max_value=9, 
    value=3, 
    step=1,
    help="Total count of active services subscribed."
)

st.sidebar.markdown("---")
st.sidebar.caption("💡 **Tip:** Adjust metrics and click 'Calculate Churn Risk Probability' below to evaluate.")

# =========================================================
# MAIN PAGE CATEGORICAL INPUTS
# =========================================================
st.subheader("📋 Customer Profile & Subscribed Services")
st.markdown("Configure categorical profile attributes and service subscriptions below:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 👤 Customer & Contract Profile")
    gender = st.selectbox("Gender", sorted(data['gender'].unique()))
    senior_citizen = st.selectbox("Senior Citizen Status", sorted(data['SeniorCitizen'].unique()), format_func=lambda x: "Yes (Senior)" if x == 1 else "No (Non-Senior)")
    partner = st.selectbox("Has Partner?", sorted(data['Partner'].unique()))
    dependents = st.selectbox("Has Dependents?", sorted(data['Dependents'].unique()))
    contract = st.selectbox("Contract Type", sorted(data['Contract'].unique()))
    paperless_billing = st.selectbox("Paperless Billing", sorted(data['PaperlessBilling'].unique()))
    payment_method = st.selectbox("Payment Method", sorted(data['PaymentMethod'].unique()))

with col2:
    st.markdown("##### 📡 Phone & Primary Connectivity")
    phone_service = st.selectbox("Phone Service", sorted(data['PhoneService'].unique()))
    multiple_lines = st.selectbox("Multiple Lines", sorted(data['MultipleLines'].unique()))
    internet_service = st.selectbox("Internet Service Type", sorted(data['InternetService'].unique()))
    online_security = st.selectbox("Online Security", sorted(data['OnlineSecurity'].unique()))
    online_backup = st.selectbox("Online Backup", sorted(data['OnlineBackup'].unique()))

with col3:
    st.markdown("##### 🛡️ Support & Add-on Services")
    device_protection = st.selectbox("Device Protection", sorted(data['DeviceProtection'].unique()))
    tech_support = st.selectbox("Tech Support", sorted(data['TechSupport'].unique()))
    streaming_tv = st.selectbox("Streaming TV", sorted(data['StreamingTV'].unique()))
    streaming_movies = st.selectbox("Streaming Movies", sorted(data['StreamingMovies'].unique()))
    is_auto_payment = st.selectbox("Auto-Payment Enrolled?", sorted(data['Is_Auto_Payment'].unique()), format_func=lambda x: "Yes (Auto)" if x == 1 else "No (Manual)")

st.markdown("<br>", unsafe_allow_html=True)

# Predict Button
predict_btn = st.button("🔮 Calculate Churn Risk Probability", use_container_width=True)

if predict_btn:
    # Construct input dataframe
    input_df = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Total_Services': total_services,
        'Is_Auto_Payment': is_auto_payment
    }])
    
    # Generate prediction & probability
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]
    prob_percent = probability * 100
    
    st.markdown("---")
    st.subheader("📊 Prediction Result & Risk Assessment")
    
    res_col1, res_col2 = st.columns([1.2, 1])
    
    with res_col1:
        # Plotly Risk Gauge Meter
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_percent,
            number={'suffix': "%", 'font': {'size': 36, 'color': '#0F172A', 'family': 'sans-serif'}},
            title={'text': "<b>Churn Probability Score</b>", 'font': {'size': 18, 'color': '#0F172A'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748B"},
                'bar': {'color': "#0F172A", 'thickness': 0.25},
                'steps': [
                    {'range': [0, 30], 'color': "#10B981"},    # Green Safe
                    {'range': [30, 60], 'color': "#F59E0B"},   # Orange Warning
                    {'range': [60, 100], 'color': "#FF4D4D"}   # Red Danger
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': prob_percent
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with res_col2:
        st.markdown("#### Status Summary")
        if prob_percent >= 60:
            st.error(f"🚨 **HIGH CHURN RISK** ({prob_percent:.1f}% Probability)")
            st.markdown("This customer profile exhibits critical churn triggers. Immediate intervention is required to prevent customer attrition.")
        elif prob_percent >= 30:
            st.warning(f"⚠️ **MODERATE CHURN RISK** ({prob_percent:.1f}% Probability)")
            st.markdown("This customer shows elevated risk indicators. Consider proactive engagement and promotional incentive offers.")
        else:
            st.success(f"✅ **LOW CHURN RISK** ({prob_percent:.1f}% Probability)")
            st.markdown("This customer profile appears highly stable with strong brand loyalty and low attrition likelihood.")

    # Retention Recommendation Panel
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("💡 Tailored Actionable Retention Strategy")
    
    recs = []
    if contract == 'Month-to-month':
        recs.append("📜 **Contract Incentive:** Customer is on a Month-to-Month plan. Offer a 10-15% discount on upgrading to an Annual 1-Year or 2-Year Contract.")
    if internet_service == 'Fiber optic' and tech_support == 'No':
        recs.append("🛡️ **Support Bundle:** Fiber Optic subscribers without Tech Support churn heavily. Offer 3 months of free Tech Support & Security add-ons.")
    if is_auto_payment == 0:
        recs.append("💳 **Automated Payment Incentive:** Customer uses manual payments. Provide a $5 monthly bill credit for switching to automated payment methods.")
    if tenure < 12 and monthly_charges > 70:
        recs.append("⏳ **Early Tenure Risk:** Customer is in their first 12 months with high charges. Assign a dedicated onboarding support specialist.")
        
    if not recs:
        recs.append("⭐ **Loyalty Rewards:** Customer is currently well-retained. Continue standard customer appreciation rewards and quarterly satisfaction check-ins.")
        
    for rec in recs:
        st.markdown(f"- {rec}")
else:
    st.markdown("---")
    st.info("👈 Adjust the customer metrics on the sidebar/form and click 'Calculate Churn Risk Probability' to generate results.")

