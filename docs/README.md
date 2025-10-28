# 📚 Documentation

Welcome to the Telco Customer Churn Analysis documentation! This section provides comprehensive guides and references for using and understanding this project.

## 📖 Table of Contents

- [Getting Started](getting_started.md)
- [API Reference](api_reference.md)
- [Tutorials](tutorials.md)
- [Troubleshooting](troubleshooting.md)
- [FAQ](faq.md)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/telco-churn-analysis.git
cd telco-churn-analysis

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Start Jupyter Notebook
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_exploratory_analysis.ipynb
# 2. notebooks/02_model_building.ipynb
```

### Running the Demo App

```bash
# Install Streamlit if not already installed
pip install streamlit

# Run the demo app
streamlit run app.py
```

## 📊 Project Structure

```
telco-churn-analysis/
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── CONTRIBUTING.md              # Contribution guidelines
├── data/                        # Dataset files
│   └── telco-customer-churn.csv
├── notebooks/                   # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_model_building.ipynb
├── src/                         # Source code modules
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── visualization.py
├── reports/                     # Generated reports
│   ├── sweetviz_report.html
│   └── pandas_profiling_report.html
├── visualizations/              # Saved plots and charts
├── docs/                        # Documentation files
└── .github/                     # GitHub configuration
    ├── ISSUE_TEMPLATE/
    └── workflows/
```

## 🔧 Key Components

### Data Preprocessing (`src/data_preprocessing.py`)
- Data loading and inspection
- Missing value handling
- Categorical encoding
- Train-test splitting
- SMOTE oversampling

### Model Training (`src/model_training.py`)
- Multiple ML algorithms
- Hyperparameter tuning
- Model evaluation
- Performance comparison

### Visualization (`src/visualization.py`)
- Target distribution plots
- Feature correlation heatmaps
- Feature importance analysis
- Model performance comparisons

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 85.6% | 72.1% | 85.6% | 78.2% |
| Random Forest | 84.2% | 70.8% | 84.2% | 77.0% |
| Logistic Regression | 81.8% | 68.7% | 57.1% | 62.4% |
| Naive Bayes | 75.9% | 53.0% | 77.5% | 63.0% |

## 🎯 Key Insights

### Top Predictive Features
1. **Contract Type** (0.092) - Month-to-month contracts have higher churn
2. **Tenure** (0.077) - Shorter tenure correlates with churn
3. **Online Security** (0.061) - Customers without security services churn more
4. **Tech Support** (0.059) - Lack of tech support increases churn risk
5. **Monthly Charges** (0.045) - Higher charges correlate with churn

### Business Recommendations
- Focus on contract retention strategies
- Enhance support services offerings
- Review pricing for high-risk segments
- Implement early intervention programs

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## 📞 Support

If you encounter any issues or have questions:

1. Check the [FAQ](faq.md)
2. Review the [Troubleshooting Guide](troubleshooting.md)
3. Open a [GitHub Issue](https://github.com/yourusername/telco-churn-analysis/issues)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
