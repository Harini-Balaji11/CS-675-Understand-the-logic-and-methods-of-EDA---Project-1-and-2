# Project Overview

This project involves the analysis of a telecommunications customer churn dataset to uncover insights, identify key features contributing to customer churn, and prepare data for machine learning models. The project uses various libraries for data preprocessing, visualization, and exploratory data analysis (EDA). The final output includes detailed profiling reports, visualizations, and a summary of feature importance.

# Table of Contents

Introduction

Dataset Overview

Project Structure

Key Steps

Findings and Limitations

# Introduction

Customer churn is a critical challenge for businesses, especially in highly competitive sectors like telecommunications. This project aims to analyze a customer churn dataset, identify key patterns and correlations, and provide data insights for predictive modeling.


README.md
Telco Customer Churn Analysis Project
Project Overview
This project involves the analysis of a telecommunications customer churn dataset to uncover insights, identify key features contributing to customer churn, and prepare data for machine learning models. The project uses various libraries for data preprocessing, visualization, and exploratory data analysis (EDA). The final output includes detailed profiling reports, visualizations, and a summary of feature importance.

Table of Contents
Introduction
Dataset Overview
Key Steps
Findings and Limitations
How to Run
Technologies Used
Contributors
Introduction
Customer churn is a critical challenge for businesses, especially in highly competitive sectors like telecommunications. This project aims to analyze a customer churn dataset, identify key patterns and correlations, and provide data insights for predictive modeling.

# Dataset Overview

Dataset Name: Telco Customer Churn

# Features:

Customer demographics: Gender, Senior Citizen, Partner, Dependents
Account information: Tenure, Monthly Charges, Total Charges
Services signed up for: Internet Service, Phone Service, etc.
Churn Indicator: Target variable to predict

# Key Steps

1. Data Loading and Preprocessing
   
The dataset is loaded using pandas and inspected for missing values and data types.
Key transformations:
Conversion of the TotalCharges column to numeric.
Encoding of categorical features using one-hot encoding and LabelEncoder.
Handling of missing values.

2. Exploratory Data Analysis (EDA)

Visualizations:
Histograms for numeric features.
Heatmap for correlation analysis.
Boxplots for feature relationships (e.g., Monthly Charges vs. Churn).
Statistical Insights:
Summary statistics (mean, median, standard deviation, etc.).
Identification of outliers using IQR.

3. Automated Profiling
Generated detailed profiling reports for training and testing datasets using ydata_profiling and sweetviz.

4. Feature Importance
Calculated mutual information scores for features relative to the target variable.
Visualized the top 10 most important features.

5. Train-Test Split
The dataset was split into training (80%) and testing (20%) sets using train_test_split.

# Findings and Limitations

# Findings:

Strong correlations between churn and features such as Monthly Charges and Tenure.
Some categorical features like Contract Type also play a significant role in churn prediction.

# Limitations:

Time Period Not Provided: The dataset lacks temporal information about when data was collected, limiting trend analysis.
Dynamic Changes: No additional data is provided to observe temporal or behavioral changes in customer churn.

