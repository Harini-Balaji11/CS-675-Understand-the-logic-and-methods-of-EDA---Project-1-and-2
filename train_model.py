"""
Model Training and Persistence Script
Train models and save them for production use
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
from datetime import datetime

def load_and_preprocess_data(file_path='data/telco-customer-churn.csv'):
    """Load and preprocess the dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print("Preprocessing data...")
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Handle missing values
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df, label_encoders

def train_models(df, label_encoders):
    """Train multiple models and return the best one"""
    print("Preparing data for training...")
    
    # Separate features and target
    X = df.drop(['Churn', 'customerID'], axis=1)
    y = df['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training models...")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        if name == 'Naive Bayes':
            model.fit(X_train_smote, y_train_smote)
        else:
            model.fit(X_train_scaled, y_train_smote)
        
        # Make predictions
        if name == 'Naive Bayes':
            y_pred = model.predict(X_train_smote)
            y_test_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        train_accuracy = accuracy_score(y_train_smote, y_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        results[name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
        
        trained_models[name] = model
        
        print(f"{name} - Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_f1'])
    best_model = trained_models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Test F1 Score: {results[best_model_name]['test_f1']:.4f}")
    
    return best_model, scaler, label_encoders, X.columns, results

def save_model_and_artifacts(model, scaler, label_encoders, feature_names, results):
    """Save the trained model and all artifacts"""
    print("Saving model and artifacts...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    with open('models/churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save label encoders
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save feature names
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('models/model_results.csv')
    
    # Save model info
    model_info = {
        'model_type': type(model).__name__,
        'training_date': datetime.now().isoformat(),
        'feature_count': len(feature_names),
        'best_f1_score': max(results.values(), key=lambda x: x['test_f1'])['test_f1']
    }
    
    with open('models/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print("Model and artifacts saved successfully!")

def load_model_and_artifacts():
    """Load the trained model and artifacts"""
    try:
        with open('models/churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        return model, scaler, label_encoders, feature_names
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        return None, None, None, None

def predict_churn(model, scaler, label_encoders, feature_names, input_data):
    """Make churn prediction for new data"""
    # Convert input data to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Apply label encoding
    for col in df_input.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            df_input[col] = label_encoders[col].transform(df_input[col].astype(str))
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in df_input.columns:
            df_input[feature] = 0
    
    # Reorder columns to match training data
    df_input = df_input[feature_names]
    
    # Scale features
    df_scaled = scaler.transform(df_input)
    
    # Make prediction
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]
    
    return prediction, probability

def main():
    """Main function to train and save models"""
    print("🚀 Starting Model Training Pipeline")
    print("=" * 50)
    
    # Load and preprocess data
    df, label_encoders = load_and_preprocess_data()
    
    # Train models
    best_model, scaler, label_encoders, feature_names, results = train_models(df, label_encoders)
    
    # Save model and artifacts
    save_model_and_artifacts(best_model, scaler, label_encoders, feature_names, results)
    
    # Display results summary
    print("\n📊 Model Performance Summary:")
    print("=" * 50)
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Model saved to: models/churn_model.pkl")
    print(f"📊 Results saved to: models/model_results.csv")

if __name__ == "__main__":
    main()
