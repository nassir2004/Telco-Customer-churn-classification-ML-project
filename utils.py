import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Color Constants
COLOR_NO_CHURN = "#0066FF"   # Royal Blue (Stay)
COLOR_CHURN = "#FF4D4D"      # Crimson Red (Churn)
COLOR_ACCENT = "#10B981"     # Emerald Green
COLOR_NAVY = "#0F172A"       # Slate Navy
COLOR_BG = "#F8FAFC"         # Light Gray-Blue

CHURN_COLOR_MAP = {"No": COLOR_NO_CHURN, "Yes": COLOR_CHURN}
CHURN_LABEL_MAP = {"No": "Stayed (Retained)", "Yes": "Churned"}

@st.cache_data
def load_data():
    """Load cleaned Telco dataset with caching."""
    df = pd.read_csv('cleaned_telco_data.csv')
    return df

@st.cache_resource
def load_pipeline():
    """Load XGBoost pre-trained classification pipeline."""
    pipeline = joblib.load('churn_xgb_pipeline.pkl')
    return pipeline

def apply_custom_css():
    """Inject premium CSS styles across all Streamlit pages."""
    css = """
    <style>
    /* Global Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    
    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0066FF 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 102, 255, 0.25);
    }
    .hero-banner h1 {
        color: #FFFFFF !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.4rem !important;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: #94A3B8 !important;
        font-size: 1.05rem !important;
        margin-bottom: 0 !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 102, 255, 0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10B981;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    .metric-sub.negative {
        color: #FF4D4D;
    }
    
    /* Insight / Takeaway Card */
    .takeaway-card {
        background: #F0F7FF;
        border-left: 4px solid #0066FF;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-top: 0.6rem;
        margin-bottom: 1.5rem;
        color: #1E293B;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .takeaway-card strong {
        color: #0066FF;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #0066FF 0%, #0052CC 100%);
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(0, 102, 255, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #0052CC 0%, #003D99 100%);
        box-shadow: 0 6px 20px rgba(0, 102, 255, 0.45) !important;
        transform: translateY(-1px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        padding-left: 16px;
        padding-right: 16px;
        font-weight: 600;
        background-color: #E2E8F0;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0066FF !important;
        color: white !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_hero_banner(title, subtitle):
    """Render modern gradient hero header banner."""
    html = f"""
    <div class="hero-banner">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_metric_card(label, value, sub_text=None, is_negative=False):
    """Render stylish custom KPI metric card."""
    sub_class = "negative" if is_negative else ""
    sub_html = f'<div class="metric-sub {sub_class}">{sub_text}</div>' if sub_text else ""
    
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_takeaway(text, title="💡 Business Key Takeaway"):
    """Render styled business insight callout container."""
    html = f"""
    <div class="takeaway-card">
        <strong>{title}:</strong> {text}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def format_plotly_chart(fig, title="", height=400):
    """Apply consistent styling to Plotly Express charts."""
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>" if title else "",
            'font': {'size': 16, 'color': '#0F172A', 'family': 'sans-serif'},
            'x': 0.02,
            'y': 0.95
        },
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=50 if title else 20, b=40),
        font=dict(family='sans-serif', size=12, color='#334155'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
            font=dict(size=11)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#E2E8F0',
            zeroline=False,
            showline=True,
            linecolor='#CBD5E1'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#E2E8F0',
            zeroline=False,
            showline=True,
            linecolor='#CBD5E1'
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_family="sans-serif"
        )
    )
    return fig
