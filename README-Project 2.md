# CS-675-Understand-the-logic-and-methods-of-EDA---Project-2

# Table of Contents:

Overview

Dataset

Installation

Prerequisites

Dependencies

Setup

Exploratory Data Analysis (EDA)

Preprocessing and Feature Engineering

Model Building

Hyperparameter Tuning

Evaluation Metrics

Results

Usage

Contributing

# Overview

Customer churn, defined as customer loss, is a crucial metric for businesses, particularly in subscription-based industries like telecommunications. Retaining customers is more cost-effective than acquiring new ones, and identifying potential churners allows companies to take proactive measures.

# Objective: The goal of this project is to:

Analyze and preprocess the Telco Customer Churn dataset.
Develop machine learning models to predict customer churn.
Enhance model performance using advanced techniques such as oversampling with SMOTE and hyperparameter tuning.

# Dataset
The dataset used in this project is the Telco Customer Churn dataset, containing customer demographic information, subscription details, payment methods, and service usage statistics.

# Key Features:

# Demographic Information:

Gender, SeniorCitizen, Partner, Dependents.

# Subscription Details:

Contract type (month-to-month, one year, two years), InternetService, PaymentMethod.

# Service Usage:

Tenure, MonthlyCharges, TotalCharges.

# Target Variable:

Churn: Indicates whether the customer churned (1) or stayed (0).

# Dataset Statistics:
Shape: 7043 rows × 21 columns.

# Imbalance: 
Approximately 26% of the customers churned, leading to a significant class imbalance.

# Exploratory Data Analysis (EDA)

# Goals:

Understand the distribution of features and their relationships with the target variable.
Identify missing values, outliers, and potential transformations.

# Key Insights:

Customers with shorter tenure and higher monthly charges are more likely to churn.
Contract type plays a significant role in churn, with month-to-month contracts being more prone to churn.
Payment method and senior citizen status also influence churn.

# Visualizations:
Target Variable Distribution: Countplot of Churn to visualize class imbalance.
Feature Histograms: Histograms for all numerical features.
Correlation Heatmap: To understand relationships between numerical variables.
Boxplots: To analyze distributions of features like MonthlyCharges and Tenure against Churn.

# Preprocessing and Feature Engineering

Steps:
Data Cleaning:
Imputed missing values in TotalCharges using the median.
Handled categorical data using LabelEncoder and OneHotEncoder.

Feature Scaling:
Standardized numerical features using StandardScaler.

Oversampling:
Applied SMOTE to address class imbalance in the target variable.

Feature Importance:
Mutual Information: Identified top predictors of churn, such as Contract, tenure, MonthlyCharges.

# Model Building
# Machine Learning Models:

Logistic Regression: Interpretable and performs well on linearly separable data.
Random Forest: Handles non-linearity and provides feature importance.
XGBoost: High-performance gradient boosting algorithm.
Naive Bayes: Probabilistic model for quick baseline performance.

# Hyperparameter Tuning
Performed GridSearchCV on the following models:

Logistic Regression: Tested different C values and solvers.
Random Forest: Tuned n_estimators, max_depth, min_samples_split, and min_samples_leaf.
XGBoost: Tuned n_estimators, max_depth, learning_rate, and subsample.

# Evaluation Metrics
Evaluation is based on the following metrics:

Accuracy: Proportion of correct predictions.
Precision: Ability to avoid false positives.
Recall: Ability to identify true positives.
F1 Score: Harmonic mean of precision and recall.
Confusion Matrix: Breakdown of true positives, false positives, true negatives, and false negatives.


