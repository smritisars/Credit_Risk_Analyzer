"""
Credit Risk Analyzer - Streamlit Dashboard
==========================================

Interactive Streamlit dashboard for credit risk analysis and loan assessment.
This provides an alternative interface to the HTML dashboard with enhanced interactivity.

Author: [Your Name]
Date: October 2025

Usage:
    streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Custom imports (assuming these files exist)
try:
    from risk_utils import RiskAssessment, PortfolioAnalyzer, create_risk_report
except ImportError:
    st.warning("risk_utils.py not found. Using fallback functions.")
    RiskAssessment = None

# Page configuration
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
    }
    
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high { color: #dc3545; font-weight: bold; }
    
    .approval-approve { 
        background-color: #d4edda; 
        color: #155724; 
        padding: 1rem; 
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    
    .approval-decline { 
        background-color: #f8d7da; 
        color: #721c24; 
        padding: 1rem; 
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    
    .approval-review { 
        background-color: #fff3cd; 
        color: #856404; 
        padding: 1rem; 
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏦 Credit Risk Analyzer for Commercial Banking</h1>', unsafe_allow_html=True)

# Initialize session state
if 'risk_assessor' not in st.session_state:
    if RiskAssessment:
        st.session_state.risk_assessor = RiskAssessment()
    else:
        st.session_state.risk_assessor = None

if 'portfolio_analyzer' not in st.session_state:
    st.session_state.portfolio_analyzer = PortfolioAnalyzer() if 'PortfolioAnalyzer' in globals() else None

# Load sample data
@st.cache_data
def load_sample_data():
    """Load sample portfolio data"""
    try:
        df = pd.read_csv('portfolio_data.csv')
    except:
        # Generate sample data if file doesn't exist
        np.random.seed(42)
        size = 1000
        
        data = {
            'loan_id': [f'LC{i:06d}' for i in range(1, size + 1)],
            'loan_amount': np.random.uniform(5000, 35000, size),
            'default_prob': np.random.beta(2, 8, size),
            'interest_rate': np.random.uniform(6, 24, size),
            'loan_grade': np.random.choice(['A', 'B', 'C', 'D', 'E'], size, p=[0.3, 0.3, 0.25, 0.1, 0.05]),
            'loan_purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'other'], 
                                           size, p=[0.4, 0.3, 0.2, 0.1]),
            'annual_income': np.random.lognormal(10.5, 0.5, size),
            'credit_score': 850 - (np.random.beta(2, 8, size) * 550),
            'expected_loss': np.random.uniform(100, 5000, size)
        }
        df = pd.DataFrame(data)
    
    return df

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Analysis Type",
    ["📊 Portfolio Overview", "🎯 Loan Application Scorer", "📈 Portfolio Analytics", "⚠️ Risk Monitoring", "🎛️ Model Performance"]
)

# Load data
portfolio_data = load_sample_data()

# Portfolio Overview Page
if page == "📊 Portfolio Overview":
    st.header("Portfolio Overview Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_loans = len(portfolio_data)
    total_exposure = portfolio_data['loan_amount'].sum()
    avg_default_rate = portfolio_data['default_prob'].mean()
    total_expected_loss = portfolio_data['expected_loss'].sum()
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Loans", f"{total_loans:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Exposure", f"${total_exposure:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Avg Default Rate", f"{avg_default_rate:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Expected Loss", f"${total_expected_loss:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Grade Distribution")
        grade_dist = portfolio_data['loan_grade'].value_counts()
        fig_pie = px.pie(values=grade_dist.values, names=grade_dist.index, 
                        title="Portfolio by Risk Grade", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Default Probability Distribution")
        fig_hist = px.histogram(portfolio_data, x='default_prob', nbins=20, 
                               title="Default Probability Distribution",
                               color_discrete_sequence=['#1f4e79'])
        fig_hist.update_layout(xaxis_title="Default Probability", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Recent Applications Table
    st.subheader("Recent Loan Applications (Top 10)")
    recent_loans = portfolio_data.head(10)[['loan_id', 'loan_amount', 'loan_grade', 'default_prob', 'loan_purpose']]
    recent_loans['default_prob'] = recent_loans['default_prob'].apply(lambda x: f"{x:.1%}")
    recent_loans['loan_amount'] = recent_loans['loan_amount'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(recent_loans, use_container_width=True)

# Loan Application Scorer Page
elif page == "🎯 Loan Application Scorer":
    st.header("Loan Application Risk Scorer")
    
    # Input Form
    with st.container():
        st.subheader("Loan Application Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loan_amount = st.slider("Loan Amount ($)", 1000, 40000, 15000, step=1000)
            annual_income = st.number_input("Annual Income ($)", min_value=20000, max_value=200000, value=50000)
            dti_ratio = st.slider("Debt-to-Income Ratio (%)", 0.0, 50.0, 18.0, step=0.5)
            emp_length = st.selectbox("Employment Length (years)", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=5)
        
        with col2:
            home_ownership = st.radio("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
            loan_purpose = st.selectbox("Loan Purpose", 
                                      ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase"])
            credit_grade = st.selectbox("Credit Grade", ["A", "B", "C", "D", "E", "F", "G"], index=2)
            term = st.selectbox("Loan Term (months)", [36, 60])
        
        with col3:
            int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.5, step=0.1)
            delinq_2yrs = st.number_input("Delinquencies (last 2 years)", min_value=0, max_value=10, value=0)
            inq_last_6mths = st.number_input("Credit Inquiries (last 6 months)", min_value=0, max_value=10, value=1)
            revol_util = st.slider("Credit Utilization (%)", 0.0, 100.0, 45.0, step=1.0)
    
    # Calculate Risk Score Button
    if st.button("🎯 Calculate Risk Score", type="primary"):
        
        # Prepare loan data
        loan_data = {
            'loan_amnt': loan_amount,
            'term': term,
            'int_rate': int_rate,
            'grade': credit_grade,
            'emp_length': emp_length,
            'home_ownership': home_ownership,
            'annual_inc': annual_income,
            'purpose': loan_purpose,
            'dti': dti_ratio,
            'delinq_2yrs': delinq_2yrs,
            'inq_last_6mths': inq_last_6mths,
            'revol_util': revol_util,
            'open_acc': 8,  # Default values
            'pub_rec': 0,
            'revol_bal': annual_income * 0.2,
            'total_acc': 15
        }
        
        # Calculate risk (using fallback if model not available)
        if st.session_state.risk_assessor:
            try:
                default_prob = st.session_state.risk_assessor.predict_default_probability(loan_data)
                credit_score = st.session_state.risk_assessor.calculate_credit_score(default_prob)
                risk_grade, risk_desc = st.session_state.risk_assessor.get_risk_grade(default_prob)
                expected_loss = st.session_state.risk_assessor.calculate_expected_loss(default_prob, loan_amount)
                recommendation = st.session_state.risk_assessor.get_approval_recommendation(default_prob, loan_data)
            except:
                # Fallback calculation
                default_prob = _fallback_risk_calculation(loan_data)
                credit_score = int(850 - (default_prob * 550))
                risk_grade = "C" if default_prob < 0.25 else "D"
                risk_desc = "Medium Risk" if default_prob < 0.25 else "High Risk"
                expected_loss = default_prob * 0.45 * loan_amount
                recommendation = {'decision': 'REVIEW', 'reason': 'Risk assessment complete'}
        else:
            # Fallback calculation
            default_prob = _fallback_risk_calculation(loan_data)
            credit_score = int(850 - (default_prob * 550))
            risk_grade = "C" if default_prob < 0.25 else "D"
            risk_desc = "Medium Risk" if default_prob < 0.25 else "High Risk"
            expected_loss = default_prob * 0.45 * loan_amount
            recommendation = {'decision': 'REVIEW', 'reason': 'Risk assessment complete'}
        
        # Display Results
        st.subheader("Risk Assessment Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_class = "risk-low" if default_prob < 0.15 else "risk-medium" if default_prob < 0.30 else "risk-high"
            st.markdown(f'<div class="metric-container"><h3>Default Probability</h3><p class="{risk_class}">{default_prob:.1%}</p></div>', 
                       unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-container"><h3>Credit Score</h3><p><strong>{credit_score}</strong></p></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-container"><h3>Risk Grade</h3><p><strong>{risk_grade}</strong></p><p>{risk_desc}</p></div>', 
                       unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'<div class="metric-container"><h3>Expected Loss</h3><p><strong>${expected_loss:,.0f}</strong></p></div>', 
                       unsafe_allow_html=True)
        
        # Recommendation
        st.subheader("Loan Recommendation")
        decision_class = f"approval-{recommendation['decision'].lower()}"
        st.markdown(f'''
        <div class="{decision_class}">
            <h4>Decision: {recommendation['decision']}</h4>
            <p><strong>Reason:</strong> {recommendation['reason']}</p>
        </div>
        ''', unsafe_allow_html=True)

# Portfolio Analytics Page
elif page == "📈 Portfolio Analytics":
    st.header("Portfolio Analytics Dashboard")
    
    # Filters
    st.subheader("Portfolio Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        grade_filter = st.multiselect("Select Risk Grades", 
                                     options=['A', 'B', 'C', 'D', 'E'], 
                                     default=['A', 'B', 'C', 'D', 'E'])
    
    with col2:
        amount_range = st.slider("Loan Amount Range ($)", 
                               int(portfolio_data['loan_amount'].min()), 
                               int(portfolio_data['loan_amount'].max()), 
                               (5000, 35000))
    
    with col3:
        purpose_filter = st.multiselect("Loan Purpose", 
                                      options=portfolio_data['loan_purpose'].unique(),
                                      default=portfolio_data['loan_purpose'].unique())
    
    # Apply filters
    filtered_data = portfolio_data[
        (portfolio_data['loan_grade'].isin(grade_filter)) &
        (portfolio_data['loan_amount'].between(amount_range[0], amount_range[1])) &
        (portfolio_data['loan_purpose'].isin(purpose_filter))
    ]
    
    st.write(f"Filtered Portfolio: {len(filtered_data):,} loans")
    
    # Analytics Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk vs Loan Amount")
        fig_scatter = px.scatter(filtered_data, x='loan_amount', y='default_prob', 
                               color='loan_grade', size='expected_loss',
                               title="Risk Profile Analysis",
                               hover_data=['loan_purpose'])
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.subheader("Portfolio Composition")
        purpose_dist = filtered_data['loan_purpose'].value_counts()
        fig_bar = px.bar(x=purpose_dist.index, y=purpose_dist.values,
                        title="Loans by Purpose",
                        color_discrete_sequence=['#1f4e79'])
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Risk Analytics Table
    st.subheader("Risk Analytics Summary")
    risk_summary = filtered_data.groupby('loan_grade').agg({
        'loan_amount': ['count', 'sum', 'mean'],
        'default_prob': ['mean', 'std'],
        'expected_loss': 'sum'
    }).round(2)
    
    risk_summary.columns = ['Count', 'Total Amount', 'Avg Amount', 'Avg Default Rate', 'Std Default Rate', 'Total Expected Loss']
    st.dataframe(risk_summary, use_container_width=True)

# Risk Monitoring Page
elif page == "⚠️ Risk Monitoring":
    st.header("Risk Monitoring Dashboard")
    
    # Generate time series data
    dates = pd.date_range('2023-01-01', '2025-10-01', freq='M')
    time_series_data = []
    
    for i, date in enumerate(dates):
        # Simulate time series with some trend
        base_default_rate = 0.20 + 0.02 * np.sin(i/6) + np.random.normal(0, 0.01)
        
        month_data = {
            'date': date,
            'default_rate': max(0.1, min(0.3, base_default_rate)),
            'new_loans': np.random.randint(80, 150),
            'portfolio_value': 50000000 + i * 500000 + np.random.normal(0, 1000000),
            'auc_score': 0.70 + np.random.normal(0, 0.02)
        }
        time_series_data.append(month_data)
    
    ts_df = pd.DataFrame(time_series_data)
    
    # Time Series Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Default Rate Trends")
        fig_line1 = px.line(ts_df, x='date', y='default_rate', 
                           title="Monthly Default Rates",
                           color_discrete_sequence=['#dc3545'])
        fig_line1.add_hline(y=ts_df['default_rate'].mean(), line_dash="dash", 
                           annotation_text="Average")
        st.plotly_chart(fig_line1, use_container_width=True)
    
    with col2:
        st.subheader("Portfolio Value Growth")
        fig_line2 = px.line(ts_df, x='date', y='portfolio_value', 
                           title="Portfolio Value Over Time",
                           color_discrete_sequence=['#1f4e79'])
        st.plotly_chart(fig_line2, use_container_width=True)
    
    # Risk Alerts
    st.subheader("Risk Alerts & Warnings")
    
    current_default_rate = ts_df['default_rate'].iloc[-1]
    avg_default_rate = ts_df['default_rate'].mean()
    
    if current_default_rate > avg_default_rate * 1.1:
        st.error(f"⚠️ HIGH RISK ALERT: Current default rate ({current_default_rate:.1%}) is {((current_default_rate/avg_default_rate-1)*100):.1f}% above average")
    elif current_default_rate > avg_default_rate * 1.05:
        st.warning(f"⚠️ MEDIUM RISK: Current default rate ({current_default_rate:.1%}) is elevated")
    else:
        st.success(f"✅ NORMAL: Current default rate ({current_default_rate:.1%}) is within acceptable range")
    
    # Model Performance Monitoring
    st.subheader("Model Performance Monitoring")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_auc = ts_df['auc_score'].iloc[-1]
        st.metric("Current Model AUC", f"{current_auc:.3f}", 
                 delta=f"{current_auc - ts_df['auc_score'].iloc[-2]:.3f}")
    
    with col2:
        psi_score = 0.05  # Mock PSI score
        psi_status = "STABLE" if psi_score < 0.1 else "MONITOR" if psi_score < 0.25 else "REVALIDATE"
        st.metric("Population Stability Index", f"{psi_score:.3f}", psi_status)
    
    with col3:
        data_quality = 98.5  # Mock data quality score
        st.metric("Data Quality Score", f"{data_quality:.1f}%")

# Model Performance Page
elif page == "🎛️ Model Performance":
    st.header("Model Performance Analysis")
    
    # Load performance metrics
    try:
        with open('model_performance.json', 'r') as f:
            performance_data = json.load(f)
    except:
        # Mock performance data
        performance_data = {
            'auc_lr': 0.7406,
            'auc_rf': 0.6997,
            'gini_lr': 0.4812,
            'gini_rf': 0.3994
        }
    
    # Model Comparison
    st.subheader("Model Performance Comparison")
    
    models_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest'],
        'AUC Score': [performance_data.get('auc_lr', 0.74), performance_data.get('auc_rf', 0.70)],
        'Gini Coefficient': [performance_data.get('gini_lr', 0.48), performance_data.get('gini_rf', 0.40)]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(models_df, x='Model', y='AUC Score', 
                        title="Model AUC Comparison",
                        color='AUC Score', color_continuous_scale='Blues')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        fig_bar2 = px.bar(models_df, x='Model', y='Gini Coefficient', 
                         title="Model Gini Comparison",
                         color='Gini Coefficient', color_continuous_scale='Greens')
        st.plotly_chart(fig_bar2, use_container_width=True)
    
    # Feature Importance
    st.subheader("Feature Importance Analysis")
    
    try:
        feature_importance = pd.read_csv('feature_importance.csv')
    except:
        # Mock feature importance data
        features = ['Interest Rate', 'DTI Ratio', 'Revolving Balance', 'Annual Income', 'Loan Amount', 'Credit Grade']
        importance = [0.113, 0.103, 0.073, 0.069, 0.069, 0.060]
        feature_importance = pd.DataFrame({'feature': features, 'importance': importance})
    
    fig_importance = px.bar(feature_importance.head(10), 
                           x='importance', y='feature', 
                           orientation='h',
                           title="Top 10 Most Important Features",
                           color='importance', color_continuous_scale='Viridis')
    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Model Metrics Table
    st.subheader("Detailed Performance Metrics")
    
    metrics_data = {
        'Metric': ['AUC Score', 'Gini Coefficient', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Logistic Regression': [0.741, 0.481, 0.78, 0.75, 0.82, 0.78],
        'Random Forest': [0.700, 0.399, 0.76, 0.73, 0.79, 0.76]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)

# Helper Functions
def _fallback_risk_calculation(loan_data):
    """Fallback risk calculation when models are not available"""
    risk_score = 0.0
    
    # Interest rate factor
    risk_score += min(loan_data.get('int_rate', 10) / 25.0, 1.0) * 0.3
    
    # Grade factor
    grade_risk = {'A': 0.05, 'B': 0.1, 'C': 0.2, 'D': 0.35, 'E': 0.5, 'F': 0.65, 'G': 0.8}
    risk_score += grade_risk.get(loan_data.get('grade', 'C'), 0.2) * 0.25
    
    # DTI factor
    dti = loan_data.get('dti', 20)
    risk_score += min(dti / 40.0, 1.0) * 0.2
    
    # Employment factor
    emp_length = loan_data.get('emp_length', 5)
    if emp_length < 2:
        risk_score += 0.1
    
    # Loan to income factor
    loan_to_income = loan_data.get('loan_amnt', 15000) / loan_data.get('annual_inc', 50000)
    if loan_to_income > 0.5:
        risk_score += 0.15
    
    return min(risk_score, 0.95)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Credit Risk Analyzer v1.0 | Built with Streamlit | For Commercial Banking Applications</p>
</div>
""", unsafe_allow_html=True)
