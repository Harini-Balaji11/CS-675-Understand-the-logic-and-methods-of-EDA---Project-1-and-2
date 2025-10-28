"""
Visualization utilities for Telco Customer Churn Analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def plot_target_distribution(df, target_col='Churn'):
    """
    Plot distribution of target variable
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=target_col)
    plt.title(f"Distribution of Target Variable ({target_col})")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.show()
    
    # Print distribution percentages
    distribution = df[target_col].value_counts(normalize=True) * 100
    print(f"\nTarget Distribution:")
    for value, percentage in distribution.items():
        print(f"{value}: {percentage:.2f}%")


def plot_feature_histograms(df, numeric_cols=None):
    """
    Plot histograms for numeric features
    
    Args:
        df (pd.DataFrame): Dataset
        numeric_cols (list): List of numeric column names
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            df[col].hist(bins=20, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, numeric_cols=None, figsize=(12, 10)):
    """
    Plot correlation heatmap for numeric features
    
    Args:
        df (pd.DataFrame): Dataset
        numeric_cols (list): List of numeric column names
        figsize (tuple): Figure size
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='Spectral', 
                vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.show()


def plot_feature_vs_target(df, features, target_col='Churn'):
    """
    Plot boxplots of features vs target variable
    
    Args:
        df (pd.DataFrame): Dataset
        features (list): List of feature names to plot
        target_col (str): Name of target column
    """
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, feature in enumerate(features):
        if i < len(axes):
            sns.boxplot(data=df, x=target_col, y=feature, ax=axes[i])
            axes[i].set_title(f'{feature} vs {target_col}')
            axes[i].set_xlabel(target_col)
            axes[i].set_ylabel(feature)
    
    # Hide unused subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(X, y, top_n=10, figsize=(12, 8)):
    """
    Plot feature importance using mutual information
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        top_n (int): Number of top features to display
        figsize (tuple): Figure size
    """
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': mi_scores
    }).sort_values('importance', ascending=False)
    
    # Plot top features
    plt.figure(figsize=figsize)
    sns.barplot(data=feature_importance.head(top_n), 
                x='importance', y='feature')
    plt.title(f'Top {top_n} Most Important Features (Mutual Information)')
    plt.xlabel('Mutual Information Score')
    plt.tight_layout()
    plt.show()
    
    return feature_importance


def plot_model_performance_comparison(results_df, figsize=(12, 8)):
    """
    Plot comparison of model performance metrics
    
    Args:
        results_df (pd.DataFrame): Results dataframe with model performance
        figsize (tuple): Figure size
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        sns.barplot(data=results_df, x='Model', y=metric, ax=axes[i])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for j, v in enumerate(results_df[metric]):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution_before_after(y_original, y_smote, figsize=(12, 5)):
    """
    Plot class distribution before and after SMOTE
    
    Args:
        y_original (pd.Series): Original target distribution
        y_smote (pd.Series): SMOTE target distribution
        figsize (tuple): Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Before SMOTE
    y_original.value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
    ax1.set_title('Class Distribution Before SMOTE')
    ax1.set_xlabel('Churn')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)
    
    # After SMOTE
    pd.Series(y_smote).value_counts().plot(kind='bar', ax=ax2, color=['skyblue', 'lightcoral'])
    ax2.set_title('Class Distribution After SMOTE')
    ax2.set_xlabel('Churn')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()
