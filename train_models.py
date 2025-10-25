"""
Credit Risk Model Training Script
=================================

This script trains machine learning models for credit risk analysis using the Lending Club dataset.
It includes data preprocessing, feature engineering, model training, and performance evaluation.

Author: [Your Name]
Date: October 2025
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
from datetime import datetime

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, roc_curve, 
    precision_recall_curve, confusion_matrix, accuracy_score
)
import xgboost as xgb
import lightgbm as lgb

# Statistical Analysis
from scipy import stats
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class CreditRiskModeler:
    """
    Comprehensive credit risk modeling pipeline for commercial banking applications
    """
    
    def __init__(self, config=None):
        self.config = config or {
            'test_size': 0.3,
            'random_state': 42,
            'cv_folds': 5,
            'scoring_metric': 'roc_auc'
        }
        self.models = {}
        self.preprocessors = {}
        self.performance_metrics = {}
        
    def load_data(self, file_path='lending_club_sample.csv'):
        """Load and initial data exploration"""
        print(f"Loading data from {file_path}...")
        self.df = pd.read_csv(file_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Target distribution: {self.df['loan_status'].value_counts()}")
        return self.df
    
    def explore_data(self):
        """Comprehensive data exploration and quality assessment"""
        print("=== DATA EXPLORATION REPORT ===")
        
        # Basic statistics
        print(f"Dataset shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        print(f"Duplicate rows: {self.df.duplicated().sum()}")
        
        # Target analysis
        target_dist = self.df['loan_status'].value_counts(normalize=True)
        print(f"\\nTarget Distribution:")
        for value, pct in target_dist.items():
            print(f"  {value}: {pct:.2%}")
        
        # Feature analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        print(f"\\nFeature Types:")
        print(f"  Numeric: {len(numeric_cols)} columns")
        print(f"  Categorical: {len(categorical_cols)} columns")
        
        return {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'target_distribution': target_dist.to_dict(),
            'numeric_features': list(numeric_cols),
            'categorical_features': list(categorical_cols)
        }
    
    def preprocess_data(self):
        """Advanced feature engineering and preprocessing"""
        print("Starting feature engineering...")
        
        data = self.df.copy()
        
        # Handle missing values
        print("  - Handling missing values...")
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Feature engineering
        print("  - Creating derived features...")
        
        # Financial ratios
        data['loan_to_income_ratio'] = data['loan_amnt'] / (data['annual_inc'] + 1)
        data['dti_category'] = pd.cut(data['dti'], bins=[0, 10, 20, 30, 100], 
                                     labels=['Low', 'Medium', 'High', 'Very High'])
        data['income_category'] = pd.cut(data['annual_inc'], 
                                       bins=[0, 40000, 70000, 100000, np.inf],
                                       labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Credit behavior features
        data['credit_utilization'] = data['revol_util'] / 100.0
        data['credit_history_length'] = data['total_acc'] - data['open_acc']
        data['inquiries_per_account'] = data['inq_last_6mths'] / (data['open_acc'] + 1)
        
        # Risk indicators
        data['has_delinquencies'] = (data['delinq_2yrs'] > 0).astype(int)
        data['has_public_records'] = (data['pub_rec'] > 0).astype(int)
        data['high_utilization'] = (data['revol_util'] > 80).astype(int)
        
        # Loan characteristics
        data['high_interest_rate'] = (data['int_rate'] > data['int_rate'].quantile(0.75)).astype(int)
        data['large_loan'] = (data['loan_amnt'] > data['loan_amnt'].quantile(0.75)).astype(int)
        data['long_term_loan'] = (data['term'] == 60).astype(int)
        
        # Encode categorical variables
        print("  - Encoding categorical variables...")
        categorical_encoders = {}
        
        for col in ['grade', 'home_ownership', 'purpose', 'application_type']:
            if col in data.columns:
                le = LabelEncoder()
                data[f'{col}_encoded'] = le.fit_transform(data[col])
                categorical_encoders[col] = le
        
        # Store encoders
        self.preprocessors['label_encoders'] = categorical_encoders
        
        # Select final features
        feature_columns = [
            # Original features
            'loan_amnt', 'term', 'int_rate', 'emp_length', 'annual_inc', 'dti',
            'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
            
            # Encoded categorical features
            'grade_encoded', 'home_ownership_encoded', 'purpose_encoded', 'application_type_encoded',
            
            # Engineered features
            'loan_to_income_ratio', 'credit_utilization', 'credit_history_length', 'inquiries_per_account',
            'has_delinquencies', 'has_public_records', 'high_utilization', 
            'high_interest_rate', 'large_loan', 'long_term_loan'
        ]
        
        # Filter existing columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        X = data[available_features]
        y = data['loan_status']
        
        print(f"  - Final feature set: {len(available_features)} features")
        print(f"  - Feature engineering complete!")
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple machine learning models"""
        print("=== MODEL TRAINING ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], 
            stratify=y
        )
        
        # Scale features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.preprocessors['scaler'] = scaler
        self.X_test, self.y_test = X_test, y_test
        
        # Model configurations
        model_configs = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']},
                'use_scaled': True
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
                'use_scaled': False
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
                'use_scaled': False
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
                'use_scaled': False
            }
        }
        
        results = {}
        
        for name, config in model_configs.items():
            print(f"\\nTraining {name}...")
            
            # Select appropriate data
            if config['use_scaled']:
                X_train_model, X_test_model = X_train_scaled, X_test_scaled
            else:
                X_train_model, X_test_model = X_train, X_test
            
            # Grid search for best parameters
            grid_search = GridSearchCV(
                config['model'], config['params'], 
                cv=self.config['cv_folds'], 
                scoring=self.config['scoring_metric'],
                n_jobs=-1
            )
            
            grid_search.fit(X_train_model, y_train)
            
            # Best model
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test_model)
            y_pred_proba = best_model.predict_proba(X_test_model)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'auc_score': auc_score,
                'accuracy': accuracy,
                'predictions': y_pred_proba,
                'use_scaled_data': config['use_scaled']
            }
            
            print(f"  - Best params: {grid_search.best_params_}")
            print(f"  - AUC Score: {auc_score:.4f}")
            print(f"  - Accuracy: {accuracy:.4f}")
        
        self.models = results
        return results
    
    def evaluate_models(self):
        """Comprehensive model evaluation and comparison"""
        print("=== MODEL EVALUATION ===")
        
        evaluation_results = {}
        
        for name, model_data in self.models.items():
            print(f"\\nEvaluating {name}...")
            
            y_pred_proba = model_data['predictions']
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Classification metrics
            auc = roc_auc_score(self.y_test, y_pred_proba)
            gini = 2 * auc - 1
            
            # ROC curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            evaluation_results[name] = {
                'auc': auc,
                'gini': gini,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'confusion_matrix': cm.tolist(),
                'best_params': model_data['best_params']
            }
            
            print(f"  - AUC: {auc:.4f}")
            print(f"  - Gini: {gini:.4f}")
        
        self.performance_metrics = evaluation_results
        return evaluation_results
    
    def feature_importance_analysis(self):
        """Analyze feature importance across models"""
        print("=== FEATURE IMPORTANCE ANALYSIS ===")
        
        importance_data = {}
        feature_names = self.X_test.columns
        
        for name, model_data in self.models.items():
            model = model_data['model']
            
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                continue
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            importance_data[name] = importance_df
            
            print(f"\\n{name} - Top 10 Features:")
            print(importance_df.head(10)[['feature', 'importance']].to_string(index=False))
        
        return importance_data
    
    def save_models_and_results(self, output_dir='models/'):
        """Save trained models and results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"=== SAVING MODELS TO {output_dir} ===")
        
        # Save individual models
        for name, model_data in self.models.items():
            model_filename = f"{output_dir}model_{name.lower().replace(' ', '_')}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(model_data['model'], f)
            print(f"  - Saved {name} to {model_filename}")
        
        # Save preprocessors
        preprocessor_filename = f"{output_dir}preprocessors.pkl"
        with open(preprocessor_filename, 'wb') as f:
            pickle.dump(self.preprocessors, f)
        print(f"  - Saved preprocessors to {preprocessor_filename}")
        
        # Save performance metrics
        metrics_filename = f"{output_dir}performance_metrics.json"
        with open(metrics_filename, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        print(f"  - Saved performance metrics to {metrics_filename}")
        
        # Create model summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'dataset_shape': self.df.shape,
            'test_size': self.config['test_size'],
            'models_trained': list(self.models.keys()),
            'best_model': max(self.models.items(), key=lambda x: x[1]['auc_score'])[0],
            'feature_count': len(self.X_test.columns)
        }
        
        summary_filename = f"{output_dir}training_summary.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  - Saved training summary to {summary_filename}")
        
        return summary

def main():
    """Main training pipeline"""
    print("=== CREDIT RISK MODEL TRAINING PIPELINE ===")
    print(f"Training started at: {datetime.now()}")
    
    # Initialize modeler
    modeler = CreditRiskModeler()
    
    # Load and explore data
    modeler.load_data()
    exploration_results = modeler.explore_data()
    
    # Preprocess data
    X, y = modeler.preprocess_data()
    
    # Train models
    training_results = modeler.train_models(X, y)
    
    # Evaluate models
    evaluation_results = modeler.evaluate_models()
    
    # Feature importance
    feature_importance = modeler.feature_importance_analysis()
    
    # Save everything
    summary = modeler.save_models_and_results()
    
    print("\\n=== TRAINING COMPLETE ===")
    print(f"Best model: {summary['best_model']}")
    print(f"Models saved to: models/ directory")
    print(f"Training finished at: {datetime.now()}")

if __name__ == "__main__":
    main()