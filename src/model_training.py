"""
Machine Learning model training and evaluation utilities
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def get_models():
    """
    Get dictionary of models to train
    
    Returns:
        dict: Dictionary of model names and instances
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'Naive Bayes': GaussianNB()
    }
    return models


def get_hyperparameter_grids():
    """
    Get hyperparameter grids for tuning
    
    Returns:
        dict: Dictionary of parameter grids for each model
    """
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['liblinear', 'lbfgs']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0]
        }
    }
    return param_grids


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and return metrics
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    return metrics, y_pred


def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    """
    Plot confusion matrix for model evaluation
    
    Args:
        y_test (pd.Series): True labels
        y_pred (np.array): Predicted labels
        model_name (str): Name of the model
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()


def tune_hyperparameters(model, param_grid, X_train, y_train, cv=5, scoring='recall'):
    """
    Perform hyperparameter tuning using GridSearchCV
    
    Args:
        model: Model instance
        param_grid (dict): Parameter grid for tuning
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric
        
    Returns:
        Best estimator from grid search
    """
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, 
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best {scoring} Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def compare_models(models, X_train, y_train, X_test, y_test):
    """
    Train and compare multiple models
    
    Args:
        models (dict): Dictionary of models
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        pd.DataFrame: Comparison of model performance
    """
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        metrics, y_pred = evaluate_model(model, X_test, y_test, name)
        plot_confusion_matrix(y_test, y_pred, name)
        
        results.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score']
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1 Score', ascending=False)
    
    return results_df


def plot_model_comparison(results_df):
    """
    Plot comparison of model performance
    
    Args:
        results_df (pd.DataFrame): Results from model comparison
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        sns.barplot(data=results_df, x='Model', y=metric, ax=axes[i])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
