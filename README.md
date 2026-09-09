# 📊 Telco Customer Churn Analytics & AI Prediction Platform

[![Streamlit App](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-22C55E?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-6.6.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An enterprise-grade, end-to-end customer churn analytics platform built with **Streamlit**, **Plotly Express**, and **XGBoost**. This application empowers telecommunication business stakeholders to explore churn drivers, monitor retention KPIs, evaluate dataset trends, and predict individual customer churn risk probabilities with actionable retention strategies.

---

## 🌟 Key Features

### 1. 🏠 Executive Home Dashboard (`Home.py`)
- **Key Executive Metrics**: Live summary cards showcasing Total Active Customers (**7,043**), Overall Churn Rate (**26.54%**), Average Monthly Charges (**$64.76**), and Average Customer Tenure (**32.4 Months**).
- **High-Level Visual Summaries**:
  - Overall Churn Ratio Donut Chart (Stay vs. Churn proportion).
  - Contract Type vs. Churn Volume stacked bar charts.
  - Internet Service Impact overview.
  - Tenure vs. Monthly Charges scatter distribution.
- **Interactive Dataset Explorer**: Expandable data preview table with real-time multi-select filtering options by contract type and internet service.

### 2. 📊 Target Churn Analysis Page (`pages/1_📊_Churn_Analysis.py`)
In-depth exploratory dashboard featuring intuitive **Plotly Express** charts paired with plain-English business takeaways across 4 specialized tabs:
- **👤 Demographics Tab**: Evaluates Gender, Senior Citizen status, Partner presence, and Dependents.
- **🌐 Services & Support Tab**: Analyzes Fiber Optic vs. DSL performance, Tech Support, Online Security, and Total Subscribed Services count.
- **📜 Contracts & Billing Tab**: Investigates Month-to-Month contracts, Electronic Checks vs. Automated Payments, and Paperless Billing.
- **💰 Tenure & Financials Tab**: Visualizes continuous metrics using Boxplots and Histograms for Tenure, Monthly Charges, and Cumulative Charges.

### 3. 🔮 AI Churn Risk Predictor (`pages/2_🔮_Churn_Prediction.py`)
- **Sidebar Numerical Controls**: Dedicated number inputs for `Tenure (0-240 Mo)`, `Monthly Charges ($0-$300)`, `Total Charges ($0-$25,000)`, and a slider for `Total Services Count (0-9)`.
- **Structured Categorical Inputs**: Clean 3-column form layout categorizing Customer Demographics, Phone/Internet Connectivity, and Security/Add-on Services.
- **Interactive Risk Score Gauge**: Semi-circle **Plotly Gauge Meter** visualizing the exact probability score with color-coded risk bands (**Green: Low**, **Orange: Moderate**, **Red: High Risk**).
- **Automated Retention Recommendations**: Custom retention strategies generated dynamically based on customer risk triggers (e.g. annual contract incentives, free tech support trials, or auto-pay discounts).

---

## 📂 Project Architecture

```text
Telco Customer Churn project/
├── .streamlit/
│   └── config.toml                  # Streamlit visual theme configuration
├── pages/
│   ├── 1_📊_Churn_Analysis.py        # Target churn deep-dive analytics page
│   └── 2_🔮_Churn_Prediction.py      # Interactive AI predictor page
├── Home.py                          # Executive dashboard home page
├── utils.py                         # Shared CSS styling, metric cards & Plotly formatting
├── churn_xgb_pipeline.pkl           # Pre-trained Scikit-Learn XGBoost Pipeline artifact
├── cleaned_telco_data.csv           # Cleaned dataset (7,043 customer records)
├── Telco.ipynb                      # Data exploration & model training Jupyter notebook
└── requirements.txt                 # Exact python dependencies from cdsp environment
```

---

## 🤖 Machine Learning Workflow

The underlying prediction engine uses a pre-trained **XGBoost Classification Pipeline** saved as `churn_xgb_pipeline.pkl`.

- **Data Preprocessing Pipeline**:
  - **Numerical Features**: Standardized using `StandardScaler` (`tenure`, `MonthlyCharges`, `TotalCharges`, `Total_Services`).
  - **Categorical Features**: One-Hot Encoded (`OneHotEncoder(drop='first')`) for gender, partner status, contract types, internet services, and payment methods.
- **Model Architecture**: Tuned `XGBClassifier` optimized for imbalanced classification.
- **Performance Focus**: Model parameters were tuned to prioritize **High Recall (~80.85% on Churn class)**, ensuring at-risk customers are proactively flagged before departing.

---

## ⚡ How to Run Locally

### Prerequisites
Make sure you have [Conda](https://docs.conda.io/) installed on your machine.

### Step-by-Step Instructions

1. **Open your Terminal or PowerShell**:
   Navigate to the project root directory:
   ```powershell
   cd "e:\Work\courses\CDSP\Telco Customer Churn project"
   ```

2. **Activate the `cdsp` Conda Environment**:
   ```powershell
   conda activate cdsp
   ```

3. **Install Dependencies (If setting up a fresh environment)**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit Application**:
   ```powershell
   streamlit run Home.py
   ```
   *Alternative direct execution path:*
   ```powershell
   & "C:\myanaconda\envs\cdsp\python.exe" -m streamlit run Home.py
   ```

5. **Access the Web App**:
   Open your browser and navigate to: `http://localhost:8501`

---

## 🛠️ Technology Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.11+** | Core Programming Language |
| **Streamlit 1.57.0** | Interactive Web Application Framework |
| **XGBoost 3.4.1** | Gradient Boosted Machine Learning Pipeline |
| **Scikit-Learn 1.8.0** | Model Preprocessing & Feature Encoding |
| **Plotly Express 6.6.0** | Dynamic Interactive Data Visualizations |
| **Pandas 3.0.0** | Data Manipulation & Processing |
| **Joblib 1.5.3** | Machine Learning Model Serialization |

---

## 💡 Key Business Retention Takeaways

1. **Contract Strategy**: Month-to-Month contract holders experience a **42.7% churn rate**. Transitioning users to 1-Year or 2-Year contracts reduces churn to under 12%.
2. **Support Services**: Providing **Tech Support** and **Online Security** add-ons drops churn from **41.6% down to 15.2%**.
3. **Payment Automation**: Customers using automated bank transfers or credit card auto-payments churn half as often (**15.9%**) as manual payers (**34.5%**).

---

Developed for the **CDSP Telco Customer Churn Project**.
