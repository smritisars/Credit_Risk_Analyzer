# Create utility functions for the dashboard
import pandas as pd
import numpy as np

# Create risk scoring functions
def calculate_credit_score(default_prob):
    """Convert default probability to credit score (300-850 range)"""
    # Higher probability of default = lower credit score
    score = 850 - (default_prob * 550)
    return max(300, min(850, int(score)))

def get_risk_grade(default_prob):
    """Assign risk grade based on default probability"""
    if default_prob < 0.1:
        return "A", "Low Risk"
    elif default_prob < 0.2:
        return "B", "Low-Medium Risk"
    elif default_prob < 0.3:
        return "C", "Medium Risk"
    elif default_prob < 0.5:
        return "D", "Medium-High Risk"
    else:
        return "E", "High Risk"

def calculate_expected_loss(pd_prob, lgd=0.45, ead_factor=1.0, loan_amount=1000):
    """Calculate Expected Loss = PD * LGD * EAD"""
    ead = loan_amount * ead_factor
    expected_loss = pd_prob * lgd * ead
    return expected_loss

# Create sample portfolio data for dashboard
np.random.seed(42)
portfolio_size = 1000

portfolio_data = {
    'loan_id': [f'LC{i:06d}' for i in range(1, portfolio_size + 1)],
    'loan_amount': np.random.uniform(5000, 35000, portfolio_size),
    'default_prob': np.random.beta(2, 8, portfolio_size),  # Skewed towards lower default rates
    'interest_rate': np.random.uniform(6, 24, portfolio_size),
    'loan_grade': np.random.choice(['A', 'B', 'C', 'D', 'E'], portfolio_size, p=[0.3, 0.3, 0.25, 0.1, 0.05]),
    'loan_purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'other'], 
                                   portfolio_size, p=[0.4, 0.3, 0.2, 0.1]),
    'annual_income': np.random.lognormal(10.5, 0.5, portfolio_size),
    'employment_length': np.random.randint(0, 11, portfolio_size),
    'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], portfolio_size, p=[0.4, 0.2, 0.4])
}

# Calculate derived metrics
portfolio_df = pd.DataFrame(portfolio_data)
portfolio_df['credit_score'] = portfolio_df['default_prob'].apply(calculate_credit_score)
portfolio_df['risk_grade'], portfolio_df['risk_description'] = zip(*portfolio_df['default_prob'].apply(get_risk_grade))
portfolio_df['expected_loss'] = portfolio_df.apply(lambda x: calculate_expected_loss(
    x['default_prob'], loan_amount=x['loan_amount']), axis=1)
portfolio_df['monthly_payment'] = (portfolio_df['loan_amount'] * (portfolio_df['interest_rate']/100/12)) / \
                                 (1 - (1 + portfolio_df['interest_rate']/100/12)**(-36))

# Save portfolio data
portfolio_df.to_csv('portfolio_data.csv', index=False)

# Create risk summary statistics
risk_summary = {
    'total_loans': len(portfolio_df),
    'total_exposure': portfolio_df['loan_amount'].sum(),
    'avg_default_prob': portfolio_df['default_prob'].mean(),
    'total_expected_loss': portfolio_df['expected_loss'].sum(),
    'high_risk_count': len(portfolio_df[portfolio_df['default_prob'] > 0.3]),
    'avg_credit_score': portfolio_df['credit_score'].mean(),
    'grade_distribution': portfolio_df['risk_grade'].value_counts().to_dict()
}

# Save risk summary
import json
with open('risk_summary.json', 'w') as f:
    json.dump(risk_summary, f, indent=2)

print("Portfolio analysis complete!")
print(f"Portfolio size: {len(portfolio_df)} loans")
print(f"Total exposure: ${portfolio_df['loan_amount'].sum():,.2f}")
print(f"Average default probability: {portfolio_df['default_prob'].mean():.2%}")
print(f"Total expected loss: ${portfolio_df['expected_loss'].sum():,.2f}")
print("\nRisk Grade Distribution:")
print(portfolio_df['risk_grade'].value_counts())

# Create time series data for performance monitoring
dates = pd.date_range('2023-01-01', '2025-10-10', freq='M')
time_series_data = []

for date in dates:
    # Simulate monthly performance metrics
    month_data = {
        'date': date.strftime('%Y-%m-%d'),
        'new_loans': np.random.randint(80, 150),
        'avg_default_rate': np.random.uniform(0.15, 0.25),
        'portfolio_value': np.random.uniform(50000000, 80000000),
        'auc_score': np.random.uniform(0.68, 0.75),
        'gini_coefficient': np.random.uniform(0.36, 0.50)
    }
    time_series_data.append(month_data)

time_series_df = pd.DataFrame(time_series_data)
time_series_df.to_csv('performance_trends.csv', index=False)

print("\nTime series performance data created!")
print("Files created:")
print("- portfolio_data.csv")
print("- risk_summary.json") 
print("- performance_trends.csv")
