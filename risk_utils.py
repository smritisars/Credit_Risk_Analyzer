"""
Risk Assessment Utilities
========================

Utility functions for credit risk assessment, scoring, and portfolio analysis.
These functions support the credit risk analyzer dashboard and model predictions.

Author: [Your Name]
Date: October 2025
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RiskAssessment:
    """
    Credit risk assessment and scoring utilities for commercial banking applications
    """
    
    def __init__(self, model_path='models/', use_model='Random Forest'):
        self.model_path = model_path
        self.use_model = use_model
        self.load_models_and_preprocessors()
        
    def load_models_and_preprocessors(self):
        """Load trained models and preprocessing components"""
        try:
            # Load models
            with open(f'{self.model_path}model_logistic_regression.pkl', 'rb') as f:
                self.lr_model = pickle.load(f)
            
            with open(f'{self.model_path}model_random_forest.pkl', 'rb') as f:
                self.rf_model = pickle.load(f)
            
            # Load preprocessors
            with open(f'{self.model_path}preprocessors.pkl', 'rb') as f:
                self.preprocessors = pickle.load(f)
            
            print("Models and preprocessors loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"Warning: Could not load models - {e}")
            print("Using fallback scoring method...")
            self.lr_model = None
            self.rf_model = None
            self.preprocessors = None
    
    def preprocess_application(self, loan_data):
        """
        Preprocess loan application data for model prediction
        
        Parameters:
        -----------
        loan_data : dict
            Dictionary containing loan application features
            
        Returns:
        --------
        pd.DataFrame : Preprocessed feature vector
        """
        
        # Create DataFrame from input
        df = pd.DataFrame([loan_data])
        
        # Feature engineering (matching training pipeline)
        df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)
        df['credit_utilization'] = df.get('revol_util', 50) / 100.0
        df['credit_history_length'] = df.get('total_acc', 20) - df.get('open_acc', 10)
        df['inquiries_per_account'] = df.get('inq_last_6mths', 1) / (df.get('open_acc', 10) + 1)
        
        # Risk indicators
        df['has_delinquencies'] = (df.get('delinq_2yrs', 0) > 0).astype(int)
        df['has_public_records'] = (df.get('pub_rec', 0) > 0).astype(int)
        df['high_utilization'] = (df.get('revol_util', 50) > 80).astype(int)
        df['high_interest_rate'] = (df['int_rate'] > 15).astype(int)
        df['large_loan'] = (df['loan_amnt'] > 25000).astype(int)
        df['long_term_loan'] = (df['term'] == 60).astype(int)
        
        # Encode categorical variables
        if self.preprocessors and 'label_encoders' in self.preprocessors:
            encoders = self.preprocessors['label_encoders']
            
            for col, encoder in encoders.items():
                if col in df.columns:
                    try:
                        df[f'{col}_encoded'] = encoder.transform(df[col])
                    except:
                        # Handle unknown categories
                        df[f'{col}_encoded'] = 0
        else:
            # Fallback encoding
            grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
            home_mapping = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2}
            purpose_mapping = {'debt_consolidation': 0, 'credit_card': 1, 'home_improvement': 2, 'other': 3}
            
            df['grade_encoded'] = df['grade'].map(grade_mapping).fillna(2)
            df['home_ownership_encoded'] = df['home_ownership'].map(home_mapping).fillna(0)
            df['purpose_encoded'] = df['purpose'].map(purpose_mapping).fillna(3)
            df['application_type_encoded'] = 0  # Default to Individual
        
        # Select features (must match training features)
        feature_columns = [
            'loan_amnt', 'term', 'int_rate', 'emp_length', 'annual_inc', 'dti',
            'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
            'grade_encoded', 'home_ownership_encoded', 'purpose_encoded', 'application_type_encoded',
            'loan_to_income_ratio', 'credit_utilization', 'credit_history_length', 'inquiries_per_account',
            'has_delinquencies', 'has_public_records', 'high_utilization', 
            'high_interest_rate', 'large_loan', 'long_term_loan'
        ]
        
        # Fill missing features with defaults
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        return df[feature_columns]
    
    def predict_default_probability(self, loan_data):
        """
        Predict default probability for a loan application
        
        Parameters:
        -----------
        loan_data : dict
            Loan application data
            
        Returns:
        --------
        float : Default probability (0-1)
        """
        
        if self.rf_model is None:
            # Fallback scoring based on rules
            return self._fallback_risk_score(loan_data)
        
        try:
            # Preprocess data
            X = self.preprocess_application(loan_data)
            
            # Use Random Forest by default
            if self.use_model == 'Logistic Regression' and self.lr_model:
                # Scale features for logistic regression
                if 'scaler' in self.preprocessors:
                    X_scaled = self.preprocessors['scaler'].transform(X)
                    default_prob = self.lr_model.predict_proba(X_scaled)[0, 0]  # Probability of class 0 (default)
                else:
                    default_prob = self.lr_model.predict_proba(X)[0, 0]
            else:
                default_prob = self.rf_model.predict_proba(X)[0, 0]  # Probability of class 0 (default)
            
            # Convert to default probability (0 = default, 1 = fully paid in our encoding)
            return 1 - default_prob  # Return probability of default
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_risk_score(loan_data)
    
    def _fallback_risk_score(self, loan_data):
        """Fallback risk scoring when models are not available"""
        
        # Rule-based scoring
        risk_score = 0.0
        
        # Interest rate factor (higher rate = higher risk)
        risk_score += min(loan_data.get('int_rate', 10) / 25.0, 1.0) * 0.3
        
        # Grade factor
        grade_risk = {'A': 0.05, 'B': 0.1, 'C': 0.2, 'D': 0.35, 'E': 0.5, 'F': 0.65, 'G': 0.8}
        risk_score += grade_risk.get(loan_data.get('grade', 'C'), 0.2) * 0.25
        
        # DTI factor
        dti = loan_data.get('dti', 20)
        risk_score += min(dti / 40.0, 1.0) * 0.2
        
        # Employment length factor
        emp_length = loan_data.get('emp_length', 5)
        if emp_length < 2:
            risk_score += 0.1
        
        # Loan amount factor
        loan_to_income = loan_data.get('loan_amnt', 15000) / loan_data.get('annual_inc', 50000)
        if loan_to_income > 0.5:
            risk_score += 0.15
        
        return min(risk_score, 0.95)  # Cap at 95%
    
    def calculate_credit_score(self, default_prob):
        """
        Convert default probability to credit score (300-850 range)
        
        Parameters:
        -----------
        default_prob : float
            Default probability (0-1)
            
        Returns:
        --------
        int : Credit score (300-850)
        """
        # Inverse relationship: higher default prob = lower credit score
        score = 850 - (default_prob * 550)
        return max(300, min(850, int(score)))
    
    def get_risk_grade(self, default_prob):
        """
        Assign risk grade based on default probability
        
        Parameters:
        -----------
        default_prob : float
            Default probability (0-1)
            
        Returns:
        --------
        tuple : (grade, description)
        """
        if default_prob < 0.05:
            return "A", "Excellent - Very Low Risk"
        elif default_prob < 0.10:
            return "B", "Good - Low Risk"
        elif default_prob < 0.20:
            return "C", "Fair - Medium Risk"
        elif default_prob < 0.35:
            return "D", "Poor - High Risk"
        else:
            return "E", "Very Poor - Very High Risk"
    
    def calculate_expected_loss(self, default_prob, loan_amount, lgd=0.45, ead_factor=1.0):
        """
        Calculate Expected Loss = PD * LGD * EAD
        
        Parameters:
        -----------
        default_prob : float
            Probability of Default
        loan_amount : float
            Loan amount
        lgd : float
            Loss Given Default (default: 45%)
        ead_factor : float
            Exposure at Default factor
            
        Returns:
        --------
        float : Expected loss amount
        """
        ead = loan_amount * ead_factor
        expected_loss = default_prob * lgd * ead
        return expected_loss
    
    def get_approval_recommendation(self, default_prob, loan_data):
        """
        Provide loan approval recommendation based on risk assessment
        
        Parameters:
        -----------
        default_prob : float
            Default probability
        loan_data : dict
            Loan application data
            
        Returns:
        --------
        dict : Recommendation with reasoning
        """
        
        # Risk thresholds
        if default_prob < 0.15:
            decision = "APPROVE"
            confidence = "High"
            reason = "Low risk profile with strong credit indicators"
        elif default_prob < 0.25:
            decision = "APPROVE"
            confidence = "Medium"
            reason = "Acceptable risk level, monitor closely"
        elif default_prob < 0.40:
            decision = "REVIEW"
            confidence = "Low"
            reason = "Elevated risk - requires manual review"
        else:
            decision = "DECLINE"
            confidence = "High"
            reason = "High risk of default"
        
        # Additional considerations
        considerations = []
        
        if loan_data.get('dti', 0) > 30:
            considerations.append("High debt-to-income ratio")
        
        if loan_data.get('delinq_2yrs', 0) > 0:
            considerations.append("Recent delinquencies")
        
        if loan_data.get('emp_length', 10) < 2:
            considerations.append("Limited employment history")
        
        loan_to_income = loan_data.get('loan_amnt', 0) / loan_data.get('annual_inc', 1)
        if loan_to_income > 0.5:
            considerations.append("Large loan relative to income")
        
        return {
            'decision': decision,
            'confidence': confidence,
            'reason': reason,
            'considerations': considerations,
            'default_probability': default_prob,
            'risk_level': self.get_risk_grade(default_prob)[1]
        }

class PortfolioAnalyzer:
    """
    Portfolio-level risk analysis and monitoring utilities
    """
    
    def __init__(self, portfolio_data=None):
        if portfolio_data is not None:
            self.portfolio = pd.read_csv(portfolio_data) if isinstance(portfolio_data, str) else portfolio_data
        else:
            self.portfolio = self._generate_sample_portfolio()
    
    def _generate_sample_portfolio(self, size=1000):
        """Generate sample portfolio for demonstration"""
        np.random.seed(42)
        
        data = {
            'loan_id': [f'LC{i:06d}' for i in range(1, size + 1)],
            'loan_amount': np.random.uniform(5000, 35000, size),
            'default_prob': np.random.beta(2, 8, size),
            'interest_rate': np.random.uniform(6, 24, size),
            'loan_grade': np.random.choice(['A', 'B', 'C', 'D', 'E'], size, p=[0.3, 0.3, 0.25, 0.1, 0.05]),
            'loan_purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'other'], 
                                           size, p=[0.4, 0.3, 0.2, 0.1]),
            'annual_income': np.random.lognormal(10.5, 0.5, size),
            'origination_date': pd.date_range('2023-01-01', '2025-09-01', periods=size)
        }
        
        portfolio = pd.DataFrame(data)
        
        # Calculate derived metrics
        portfolio['expected_loss'] = portfolio['default_prob'] * 0.45 * portfolio['loan_amount']
        portfolio['monthly_payment'] = (portfolio['loan_amount'] * (portfolio['interest_rate']/100/12)) / \
                                     (1 - (1 + portfolio['interest_rate']/100/12)**(-36))
        
        return portfolio
    
    def calculate_portfolio_metrics(self):
        """Calculate comprehensive portfolio risk metrics"""
        
        metrics = {
            'total_loans': len(self.portfolio),
            'total_exposure': self.portfolio['loan_amount'].sum(),
            'average_loan_size': self.portfolio['loan_amount'].mean(),
            'weighted_avg_default_rate': np.average(
                self.portfolio['default_prob'], 
                weights=self.portfolio['loan_amount']
            ),
            'total_expected_loss': self.portfolio['expected_loss'].sum(),
            'expected_loss_rate': self.portfolio['expected_loss'].sum() / self.portfolio['loan_amount'].sum(),
            'high_risk_concentration': len(self.portfolio[self.portfolio['default_prob'] > 0.3]) / len(self.portfolio),
            'avg_interest_rate': np.average(
                self.portfolio['interest_rate'], 
                weights=self.portfolio['loan_amount']
            )
        }
        
        return metrics
    
    def risk_distribution_analysis(self):
        """Analyze risk distribution across the portfolio"""
        
        # Risk buckets
        risk_buckets = pd.cut(
            self.portfolio['default_prob'], 
            bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        distribution = self.portfolio.groupby(risk_buckets).agg({
            'loan_amount': ['count', 'sum', 'mean'],
            'default_prob': 'mean',
            'expected_loss': 'sum'
        }).round(2)
        
        return distribution
    
    def vintage_analysis(self):
        """Analyze performance by loan vintage (origination period)"""
        
        self.portfolio['vintage'] = pd.to_datetime(self.portfolio['origination_date']).dt.to_period('Q')
        
        vintage_stats = self.portfolio.groupby('vintage').agg({
            'loan_amount': ['count', 'sum'],
            'default_prob': 'mean',
            'expected_loss': 'sum',
            'interest_rate': 'mean'
        }).round(3)
        
        return vintage_stats
    
    def concentration_analysis(self):
        """Analyze portfolio concentrations by various dimensions"""
        
        concentrations = {}
        
        # Purpose concentration
        concentrations['by_purpose'] = self.portfolio.groupby('loan_purpose').agg({
            'loan_amount': ['sum', 'count'],
            'default_prob': 'mean'
        }).round(2)
        
        # Grade concentration
        concentrations['by_grade'] = self.portfolio.groupby('loan_grade').agg({
            'loan_amount': ['sum', 'count'],
            'default_prob': 'mean'
        }).round(2)
        
        # Size concentration
        size_buckets = pd.cut(
            self.portfolio['loan_amount'],
            bins=[0, 10000, 20000, 30000, float('inf')],
            labels=['<10k', '10k-20k', '20k-30k', '>30k']
        )
        
        concentrations['by_size'] = self.portfolio.groupby(size_buckets).agg({
            'loan_amount': ['count', 'sum'],
            'default_prob': 'mean'
        }).round(2)
        
        return concentrations

def create_risk_report(loan_data, output_format='dict'):
    """
    Generate comprehensive risk assessment report for a loan application
    
    Parameters:
    -----------
    loan_data : dict
        Loan application data
    output_format : str
        Output format ('dict', 'json', 'html')
        
    Returns:
    --------
    Risk assessment report in specified format
    """
    
    # Initialize risk assessor
    assessor = RiskAssessment()
    
    # Perform risk assessment
    default_prob = assessor.predict_default_probability(loan_data)
    credit_score = assessor.calculate_credit_score(default_prob)
    risk_grade, risk_description = assessor.get_risk_grade(default_prob)
    expected_loss = assessor.calculate_expected_loss(default_prob, loan_data.get('loan_amnt', 0))
    recommendation = assessor.get_approval_recommendation(default_prob, loan_data)
    
    # Create report
    report = {
        'loan_application': loan_data,
        'risk_assessment': {
            'default_probability': round(default_prob * 100, 2),  # As percentage
            'credit_score': credit_score,
            'risk_grade': risk_grade,
            'risk_description': risk_description,
            'expected_loss': round(expected_loss, 2)
        },
        'recommendation': recommendation,
        'assessment_date': datetime.now().isoformat(),
        'model_version': '1.0'
    }
    
    if output_format == 'json':
        return json.dumps(report, indent=2)
    elif output_format == 'html':
        return generate_html_report(report)
    else:
        return report

def generate_html_report(report):
    """Generate HTML formatted risk assessment report"""
    
    html = f'''
    <div class="risk-report">
        <h2>Credit Risk Assessment Report</h2>
        <p><strong>Assessment Date:</strong> {report['assessment_date']}</p>
        
        <h3>Loan Application Details</h3>
        <ul>
            <li><strong>Loan Amount:</strong> ${report['loan_application'].get('loan_amnt', 0):,.2f}</li>
            <li><strong>Annual Income:</strong> ${report['loan_application'].get('annual_inc', 0):,.2f}</li>
            <li><strong>Interest Rate:</strong> {report['loan_application'].get('int_rate', 0):.2f}%</li>
            <li><strong>Credit Grade:</strong> {report['loan_application'].get('grade', 'N/A')}</li>
        </ul>
        
        <h3>Risk Assessment Results</h3>
        <div class="risk-metrics">
            <p><strong>Default Probability:</strong> {report['risk_assessment']['default_probability']:.2f}%</p>
            <p><strong>Credit Score:</strong> {report['risk_assessment']['credit_score']}</p>
            <p><strong>Risk Grade:</strong> {report['risk_assessment']['risk_grade']}</p>
            <p><strong>Risk Level:</strong> {report['risk_assessment']['risk_description']}</p>
            <p><strong>Expected Loss:</strong> ${report['risk_assessment']['expected_loss']:.2f}</p>
        </div>
        
        <h3>Recommendation</h3>
        <div class="recommendation {report['recommendation']['decision'].lower()}">
            <p><strong>Decision:</strong> {report['recommendation']['decision']}</p>
            <p><strong>Confidence:</strong> {report['recommendation']['confidence']}</p>
            <p><strong>Reason:</strong> {report['recommendation']['reason']}</p>
        </div>
    </div>
    '''
    
    return html

if __name__ == "__main__":
    # Example usage
    sample_application = {
        'loan_amnt': 15000,
        'term': 36,
        'int_rate': 12.5,
        'grade': 'B',
        'emp_length': 5,
        'home_ownership': 'RENT',
        'annual_inc': 55000,
        'purpose': 'debt_consolidation',
        'dti': 18.2,
        'delinq_2yrs': 0,
        'inq_last_6mths': 1,
        'open_acc': 8,
        'pub_rec': 0,
        'revol_bal': 12000,
        'revol_util': 45.5,
        'total_acc': 15
    }
    
    # Generate risk assessment
    report = create_risk_report(sample_application)
    print("=== CREDIT RISK ASSESSMENT REPORT ===")
    print(json.dumps(report, indent=2))
