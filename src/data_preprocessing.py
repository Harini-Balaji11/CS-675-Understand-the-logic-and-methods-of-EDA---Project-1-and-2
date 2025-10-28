"""
Data preprocessing utilities for Telco Customer Churn Analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_and_inspect_data(file_path):
    """
    Load the dataset and perform initial inspection
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(file_path)
    
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nFirst 5 rows:")
    print(df.head())
    
    return df


def preprocess_data(df):
    """
    Preprocess the dataset for machine learning
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Convert TotalCharges to numeric
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df_processed['TotalCharges'] = imputer.fit_transform(df_processed[['TotalCharges']])
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    return df_processed, label_encoders


def split_and_balance_data(df, target_col='Churn', test_size=0.2, random_state=42):
    """
    Split data into train/test and apply SMOTE for balancing
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        target_col (str): Name of target column
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, X_train_smote, y_train_smote
    """
    # Separate features and target
    X = df.drop([target_col, 'customerID'], axis=1)
    y = df[target_col]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Apply SMOTE to training data
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"Original training set shape: {X_train.shape}")
    print(f"SMOTE training set shape: {X_train_smote.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Original class distribution: {y_train.value_counts().to_dict()}")
    print(f"SMOTE class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, X_train_smote, y_train_smote


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: Scaled training and test features
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler
