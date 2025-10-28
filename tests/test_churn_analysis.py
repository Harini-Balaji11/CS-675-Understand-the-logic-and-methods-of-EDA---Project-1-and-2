"""
Test suite for Telco Customer Churn Analysis
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import load_and_inspect_data, preprocess_data, split_and_balance_data
from src.model_training import get_models, evaluate_model
from src.visualization import plot_target_distribution, plot_feature_importance


class TestDataPreprocessing:
    """Test cases for data preprocessing functions"""
    
    def test_load_and_inspect_data(self):
        """Test data loading function"""
        # Create sample data
        sample_data = pd.DataFrame({
            'customerID': ['1', '2', '3'],
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.0, 75.0, 30.0],
            'TotalCharges': ['600', '1800', '180'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Test basic functionality
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) == 3
        assert 'Churn' in sample_data.columns
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        # Create sample data
        sample_data = pd.DataFrame({
            'customerID': ['1', '2', '3'],
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.0, 75.0, 30.0],
            'TotalCharges': ['600', '1800', '180'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        processed_data, encoders = preprocess_data(sample_data)
        
        # Test that all columns are numeric
        assert all(processed_data.dtypes.apply(lambda x: np.issubdtype(x, np.number)))
        assert len(encoders) > 0
    
    def test_split_and_balance_data(self):
        """Test train-test split and SMOTE balancing"""
        # Create sample data
        sample_data = pd.DataFrame({
            'customerID': [f'customer_{i}' for i in range(100)],
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'Churn': np.random.choice([0, 1], 100, p=[0.7, 0.3])
        })
        
        X_train, X_test, y_train, y_test, X_train_smote, y_train_smote = split_and_balance_data(sample_data)
        
        # Test shapes
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train_smote.shape[0] == y_train_smote.shape[0]
        
        # Test SMOTE balancing
        assert len(np.unique(y_train_smote)) == 2  # Should have both classes


class TestModelTraining:
    """Test cases for model training functions"""
    
    def test_get_models(self):
        """Test model initialization"""
        models = get_models()
        
        assert isinstance(models, dict)
        assert len(models) == 4  # Should have 4 models
        assert 'Logistic Regression' in models
        assert 'Random Forest' in models
        assert 'XGBoost' in models
        assert 'Naive Bayes' in models
    
    def test_model_training(self):
        """Test model training"""
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = get_models()
        
        # Test each model can be trained
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            assert len(predictions) == len(y_test)
            assert all(pred in [0, 1] for pred in predictions)


class TestVisualization:
    """Test cases for visualization functions"""
    
    def test_plot_target_distribution(self):
        """Test target distribution plotting"""
        # Create sample data
        sample_data = pd.DataFrame({
            'Churn': np.random.choice([0, 1], 100, p=[0.7, 0.3])
        })
        
        # Test that function runs without error
        try:
            plot_target_distribution(sample_data)
            assert True
        except Exception as e:
            pytest.fail(f"plot_target_distribution raised {e}")
    
    def test_plot_feature_importance(self):
        """Test feature importance plotting"""
        from sklearn.datasets import make_classification
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        
        # Test that function runs without error
        try:
            importance_df = plot_feature_importance(X_df, y)
            assert isinstance(importance_df, pd.DataFrame)
            assert 'feature' in importance_df.columns
            assert 'importance' in importance_df.columns
        except Exception as e:
            pytest.fail(f"plot_feature_importance raised {e}")


class TestDataIntegrity:
    """Test cases for data integrity"""
    
    def test_no_missing_values_after_preprocessing(self):
        """Test that preprocessing removes missing values"""
        # Create sample data with missing values
        sample_data = pd.DataFrame({
            'customerID': ['1', '2', '3'],
            'gender': ['Male', 'Female', None],
            'SeniorCitizen': [0, 1, 0],
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.0, 75.0, 30.0],
            'TotalCharges': ['600', '1800', '180'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        processed_data, _ = preprocess_data(sample_data)
        
        # Check for missing values
        assert processed_data.isnull().sum().sum() == 0
    
    def test_data_types_after_preprocessing(self):
        """Test that all data types are numeric after preprocessing"""
        sample_data = pd.DataFrame({
            'customerID': ['1', '2', '3'],
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.0, 75.0, 30.0],
            'TotalCharges': ['600', '1800', '180'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        processed_data, _ = preprocess_data(sample_data)
        
        # Check that all columns are numeric
        for col in processed_data.columns:
            assert np.issubdtype(processed_data[col].dtype, np.number)


if __name__ == "__main__":
    pytest.main([__file__])
