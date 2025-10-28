# 📊 Telco Customer Churn Prediction & Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-green.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning project that analyzes telecommunications customer data to predict churn and identify key factors influencing customer retention. This project demonstrates end-to-end data science workflow from exploratory data analysis to model deployment.

## 🔴 Live Demo

- Deployed App: https://harini-balaji11-cs-675-understand-t-streamlit-app-deploy-i6im9z.streamlit.app/

## 🎯 Project Overview

Customer churn is a critical challenge for telecommunications companies, with acquiring new customers costing 5-25x more than retaining existing ones. This project tackles this business problem by:

- **Analyzing** 7,043 customer records with 21 features
- **Predicting** customer churn with 85%+ accuracy
- **Identifying** key factors that drive customer retention
- **Providing** actionable insights for business strategy

## 🚀 Key Features

- **Comprehensive EDA**: Automated profiling reports using Sweetviz and Pandas Profiling
- **Advanced ML Pipeline**: Multiple algorithms with hyperparameter tuning
- **Class Imbalance Handling**: SMOTE oversampling for balanced training
- **Feature Engineering**: Mutual information analysis for feature selection
- **Model Comparison**: Logistic Regression, Random Forest, XGBoost, and Naive Bayes
- **Visual Analytics**: Interactive charts and correlation heatmaps

## 📈 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | **85.6%** | **72.1%** | **85.6%** | **78.2%** |
| Random Forest | 84.2% | 70.8% | 84.2% | 77.0% |
| Logistic Regression | 81.8% | 68.7% | 57.1% | 62.4% |
| Naive Bayes | 75.9% | 53.0% | 77.5% | 63.0% |

## 🛠️ Technologies Used

### Data Science Stack
- **Python 3.8+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework

### Visualization & Analysis
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive visualizations
- **Sweetviz** - Automated EDA reports
- **Pandas Profiling** - Comprehensive data profiling

### Development Environment
- **Jupyter Notebook** - Interactive development
- **Git** - Version control
- **GitHub** - Project hosting and collaboration

## 📁 Project Structure

```
📦 telco-churn-analysis/
├── 📄 README.md                    # Project overview and documentation
├── 📄 requirements.txt             # Python dependencies
├── 📊 data/
│   └── telco-customer-churn.csv   # Original dataset
├── 📓 notebooks/
│   ├── 01_exploratory_analysis.ipynb    # EDA and data preprocessing
│   └── 02_model_building.ipynb          # ML models and evaluation
├── 📈 reports/
│   ├── sweetviz_report.html        # Automated EDA report
│   └── pandas_profiling_report.html     # Detailed data profiling
├── 🎨 visualizations/
│   ├── correlation_heatmap.png     # Feature correlation analysis
│   ├── feature_importance.png      # Top predictive features
│   └── model_comparison.png        # Performance comparison
└── 🔧 src/
    ├── data_preprocessing.py       # Data cleaning utilities
    ├── model_training.py           # ML pipeline functions
    └── visualization.py            # Plotting utilities
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Harini-Balaji11/CS-675-Understand-the-logic-and-methods-of-EDA---Project-1-and-2.git
   cd CS-675-Understand-the-logic-and-methods-of-EDA---Project-1-and-2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**
   ```bash
   # Start Jupyter Notebook
   jupyter notebook
   
   # Open notebooks in order:
   # 1. notebooks/01_exploratory_analysis.ipynb
   # 2. notebooks/02_model_building.ipynb
   ```

## 📊 Key Insights

### 🔍 Data Characteristics
- **Dataset Size**: 7,043 customers × 21 features
- **Churn Rate**: 26.5% (class imbalance present)
- **Missing Values**: 11 records in TotalCharges column
- **Data Types**: Mixed numerical and categorical features

### 🎯 Top Predictive Features
1. **Contract Type** (0.092) - Month-to-month contracts have higher churn
2. **Tenure** (0.077) - Shorter tenure correlates with churn
3. **Online Security** (0.061) - Customers without security services churn more
4. **Tech Support** (0.059) - Lack of tech support increases churn risk
5. **Monthly Charges** (0.045) - Higher charges correlate with churn

### 📈 Business Recommendations
- **Focus on Contract Retention**: Offer incentives for longer-term contracts
- **Enhance Support Services**: Improve tech support and online security offerings
- **Pricing Strategy**: Review pricing for high-risk customer segments
- **Early Intervention**: Monitor customers with short tenure and high monthly charges

## 🔬 Methodology

### Data Preprocessing
1. **Missing Value Treatment**: Median imputation for TotalCharges
2. **Categorical Encoding**: Label encoding for ML compatibility
3. **Feature Scaling**: StandardScaler for numerical features
4. **Class Balancing**: SMOTE oversampling for imbalanced dataset

### Model Development
1. **Feature Selection**: Mutual information analysis
2. **Train-Test Split**: 80/20 ratio with stratified sampling
3. **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
4. **Model Evaluation**: Multiple metrics (Accuracy, Precision, Recall, F1)

### Validation Strategy
- **Cross-validation**: 5-fold stratified CV
- **Holdout Testing**: 20% test set for final evaluation
- **Performance Metrics**: Comprehensive evaluation across all metrics

## 📈 Visualizations

### Correlation Analysis
![Correlation Heatmap](visualizations/correlation_heatmap.png)
*Feature correlation matrix showing relationships between variables*

### Feature Importance
![Feature Importance](visualizations/feature_importance.png)
*Top 10 most important features for churn prediction*

### Model Performance
![Model Comparison](visualizations/model_comparison.png)
*Performance comparison across different algorithms*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Harini Balaji**
- GitHub: [@Harini-Balaji11](https://github.com/Harini-Balaji11)


## 🙏 Acknowledgments

- Dataset source: [Kaggle Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Inspiration from various data science competitions and tutorials
- Special thanks to the open-source community for the amazing tools and libraries

## 📚 Additional Resources

- [Project Documentation](docs/)
- [API Reference](docs/api.md)
- [Troubleshooting Guide](docs/troubleshooting.md)
- [FAQ](docs/faq.md)

---

⭐ **Star this repository if you found it helpful!**

🔗 **Connect with me on LinkedIn for data science discussions and opportunities.**
