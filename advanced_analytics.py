"""
Advanced Analytics Dashboard
Generate comprehensive insights and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class ChurnAnalytics:
    """Advanced analytics class for churn analysis"""
    
    def __init__(self, df):
        self.df = df
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup plotting parameters"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_executive_summary(self):
        """Generate executive summary with key metrics"""
        total_customers = len(self.df)
        churned_customers = self.df['Churn'].sum()
        churn_rate = (churned_customers / total_customers) * 100
        
        avg_tenure = self.df['tenure'].mean()
        avg_monthly_charges = self.df['MonthlyCharges'].mean()
        
        # Calculate revenue impact
        churned_revenue = self.df[self.df['Churn'] == 1]['MonthlyCharges'].sum()
        total_revenue = self.df['MonthlyCharges'].sum()
        revenue_at_risk = (churned_revenue / total_revenue) * 100
        
        summary = {
            'Total Customers': f"{total_customers:,}",
            'Churn Rate': f"{churn_rate:.1f}%",
            'Churned Customers': f"{churned_customers:,}",
            'Average Tenure': f"{avg_tenure:.1f} months",
            'Average Monthly Charges': f"${avg_monthly_charges:.2f}",
            'Revenue at Risk': f"${churned_revenue:,.2f}",
            'Revenue Risk %': f"{revenue_at_risk:.1f}%"
        }
        
        return summary
    
    def plot_churn_distribution(self):
        """Plot churn distribution with insights"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall churn distribution
        churn_counts = self.df['Churn'].value_counts()
        axes[0, 0].pie(churn_counts.values, labels=['Retained', 'Churned'], 
                      autopct='%1.1f%%', startangle=90, colors=['#2E8B57', '#DC143C'])
        axes[0, 0].set_title('Overall Churn Distribution', fontsize=14, fontweight='bold')
        
        # Churn by contract type
        contract_churn = self.df.groupby('Contract')['Churn'].agg(['count', 'sum', 'mean']).reset_index()
        contract_churn['churn_rate'] = contract_churn['mean'] * 100
        
        bars = axes[0, 1].bar(contract_churn['Contract'], contract_churn['churn_rate'], 
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 1].set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Churn Rate (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars, contract_churn['churn_rate']):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{rate:.1f}%', ha='center', va='bottom')
        
        # Tenure distribution by churn
        churned_tenure = self.df[self.df['Churn'] == 1]['tenure']
        retained_tenure = self.df[self.df['Churn'] == 0]['tenure']
        
        axes[1, 0].hist([retained_tenure, churned_tenure], bins=20, alpha=0.7, 
                       label=['Retained', 'Churned'], color=['#2E8B57', '#DC143C'])
        axes[1, 0].set_title('Tenure Distribution by Churn Status', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Tenure (months)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Monthly charges distribution by churn
        churned_charges = self.df[self.df['Churn'] == 1]['MonthlyCharges']
        retained_charges = self.df[self.df['Churn'] == 0]['MonthlyCharges']
        
        axes[1, 1].hist([retained_charges, churned_charges], bins=20, alpha=0.7,
                       label=['Retained', 'Churned'], color=['#2E8B57', '#DC143C'])
        axes[1, 1].set_title('Monthly Charges Distribution by Churn Status', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Monthly Charges ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance_analysis(self):
        """Plot comprehensive feature importance analysis"""
        # Prepare data for feature importance
        X = self.df.drop(['Churn', 'customerID'], axis=1)
        y = self.df['Churn']
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top 10 Features', 'Feature Categories', 
                          'Service Features', 'Demographic Features'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Top 10 features
        top_features = feature_importance.head(10)
        fig.add_trace(
            go.Bar(x=top_features['importance'], y=top_features['feature'],
                   orientation='h', name='Importance', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Feature categories
        categories = {
            'Contract': ['Contract'],
            'Tenure': ['tenure'],
            'Charges': ['MonthlyCharges', 'TotalCharges'],
            'Services': ['OnlineSecurity', 'TechSupport', 'OnlineBackup', 'DeviceProtection'],
            'Demographics': ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
        }
        
        category_importance = {}
        for category, features in categories.items():
            category_importance[category] = feature_importance[
                feature_importance['feature'].isin(features)
            ]['importance'].sum()
        
        fig.add_trace(
            go.Pie(labels=list(category_importance.keys()), 
                   values=list(category_importance.values()),
                   name="Categories"),
            row=1, col=2
        )
        
        # Service features
        service_features = feature_importance[
            feature_importance['feature'].isin(['OnlineSecurity', 'TechSupport', 
                                              'OnlineBackup', 'DeviceProtection'])
        ]
        fig.add_trace(
            go.Bar(x=service_features['feature'], y=service_features['importance'],
                   name='Service Features', marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Demographic features
        demo_features = feature_importance[
            feature_importance['feature'].isin(['gender', 'SeniorCitizen', 
                                              'Partner', 'Dependents'])
        ]
        fig.add_trace(
            go.Bar(x=demo_features['feature'], y=demo_features['importance'],
                   name='Demographic Features', marker_color='lightgreen'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Comprehensive Feature Importance Analysis")
        
        return fig
    
    def plot_customer_segmentation(self):
        """Plot customer segmentation analysis"""
        # Create customer segments based on tenure and charges
        self.df['Tenure_Segment'] = pd.cut(self.df['tenure'], 
                                          bins=[0, 12, 24, 48, 72], 
                                          labels=['New (0-12m)', 'Growing (12-24m)', 
                                                'Established (24-48m)', 'Loyal (48m+)'])
        
        self.df['Charge_Segment'] = pd.cut(self.df['MonthlyCharges'], 
                                          bins=[0, 50, 80, 120], 
                                          labels=['Low ($0-50)', 'Medium ($50-80)', 'High ($80+)'])
        
        # Create segmentation matrix
        segment_matrix = self.df.groupby(['Tenure_Segment', 'Charge_Segment'])['Churn'].agg(['count', 'mean']).reset_index()
        segment_matrix['churn_rate'] = segment_matrix['mean'] * 100
        
        # Create heatmap
        pivot_table = segment_matrix.pivot(index='Tenure_Segment', 
                                         columns='Charge_Segment', 
                                         values='churn_rate')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Churn Rate (%)'}, ax=ax)
        ax.set_title('Customer Segmentation: Churn Rate by Tenure and Charges', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Monthly Charge Segment')
        ax.set_ylabel('Tenure Segment')
        
        return fig
    
    def plot_revenue_impact_analysis(self):
        """Plot revenue impact analysis"""
        # Calculate revenue metrics
        total_revenue = self.df['MonthlyCharges'].sum()
        churned_revenue = self.df[self.df['Churn'] == 1]['MonthlyCharges'].sum()
        retained_revenue = self.df[self.df['Churn'] == 0]['MonthlyCharges'].sum()
        
        # Revenue by contract type
        contract_revenue = self.df.groupby('Contract')['MonthlyCharges'].agg(['sum', 'mean', 'count']).reset_index()
        contract_revenue['revenue_share'] = (contract_revenue['sum'] / total_revenue) * 100
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Revenue distribution
        revenue_data = [retained_revenue, churned_revenue]
        labels = ['Retained Revenue', 'Churned Revenue']
        colors = ['#2E8B57', '#DC143C']
        
        axes[0, 0].pie(revenue_data, labels=labels, autopct='%1.1f%%', 
                      startangle=90, colors=colors)
        axes[0, 0].set_title('Revenue Distribution: Retained vs Churned', 
                           fontsize=14, fontweight='bold')
        
        # Revenue by contract type
        bars = axes[0, 1].bar(contract_revenue['Contract'], contract_revenue['revenue_share'],
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 1].set_title('Revenue Share by Contract Type', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Revenue Share (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, share in zip(bars, contract_revenue['revenue_share']):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{share:.1f}%', ha='center', va='bottom')
        
        # Average revenue per customer by churn status
        avg_revenue = self.df.groupby('Churn')['MonthlyCharges'].mean()
        bars = axes[1, 0].bar(['Retained', 'Churned'], avg_revenue.values,
                             color=['#2E8B57', '#DC143C'])
        axes[1, 0].set_title('Average Revenue per Customer by Churn Status', 
                           fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Average Monthly Revenue ($)')
        
        # Add value labels
        for bar, revenue in zip(bars, avg_revenue.values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'${revenue:.2f}', ha='center', va='bottom')
        
        # Revenue at risk by tenure
        tenure_revenue = self.df.groupby('tenure')['MonthlyCharges'].sum()
        churned_tenure_revenue = self.df[self.df['Churn'] == 1].groupby('tenure')['MonthlyCharges'].sum()
        
        axes[1, 1].plot(tenure_revenue.index, tenure_revenue.values, 
                       label='Total Revenue', color='blue', alpha=0.7)
        axes[1, 1].plot(churned_tenure_revenue.index, churned_tenure_revenue.values,
                       label='Churned Revenue', color='red', alpha=0.7)
        axes[1, 1].set_title('Revenue Distribution by Tenure', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Tenure (months)')
        axes[1, 1].set_ylabel('Revenue ($)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        return fig
    
    def generate_business_recommendations(self):
        """Generate actionable business recommendations"""
        # Calculate key metrics
        churn_rate = (self.df['Churn'].sum() / len(self.df)) * 100
        
        # Contract analysis
        contract_churn = self.df.groupby('Contract')['Churn'].mean() * 100
        month_to_month_churn = contract_churn.get('Month-to-month', 0)
        
        # Tenure analysis
        avg_tenure_churned = self.df[self.df['Churn'] == 1]['tenure'].mean()
        avg_tenure_retained = self.df[self.df['Churn'] == 0]['tenure'].mean()
        
        # Revenue analysis
        churned_revenue = self.df[self.df['Churn'] == 1]['MonthlyCharges'].sum()
        total_revenue = self.df['MonthlyCharges'].sum()
        revenue_at_risk = (churned_revenue / total_revenue) * 100
        
        recommendations = {
            'High Priority': [
                f"Focus on month-to-month contracts (churn rate: {month_to_month_churn:.1f}%)",
                f"Implement early intervention for customers with tenure < {avg_tenure_churned:.0f} months",
                f"Address revenue at risk: ${churned_revenue:,.2f} ({revenue_at_risk:.1f}% of total)"
            ],
            'Medium Priority': [
                "Enhance online security and tech support services",
                "Review pricing strategy for high-risk segments",
                "Implement customer satisfaction surveys"
            ],
            'Long-term': [
                "Develop predictive churn models for proactive intervention",
                "Create customer loyalty programs",
                "Invest in customer experience improvements"
            ]
        }
        
        return recommendations

def create_analytics_dashboard(df):
    """Create comprehensive analytics dashboard"""
    analytics = ChurnAnalytics(df)
    
    # Generate all analyses
    executive_summary = analytics.generate_executive_summary()
    churn_distribution_fig = analytics.plot_churn_distribution()
    feature_importance_fig = analytics.plot_feature_importance_analysis()
    segmentation_fig = analytics.plot_customer_segmentation()
    revenue_impact_fig = analytics.plot_revenue_impact_analysis()
    recommendations = analytics.generate_business_recommendations()
    
    return {
        'executive_summary': executive_summary,
        'churn_distribution': churn_distribution_fig,
        'feature_importance': feature_importance_fig,
        'segmentation': segmentation_fig,
        'revenue_impact': revenue_impact_fig,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/telco-customer-churn.csv')
    
    # Create analytics dashboard
    dashboard = create_analytics_dashboard(df)
    
    # Save plots
    dashboard['churn_distribution'].savefig('visualizations/churn_distribution.png', 
                                          dpi=300, bbox_inches='tight')
    dashboard['segmentation'].savefig('visualizations/customer_segmentation.png', 
                                    dpi=300, bbox_inches='tight')
    dashboard['revenue_impact'].savefig('visualizations/revenue_impact.png', 
                                      dpi=300, bbox_inches='tight')
    
    print("Analytics dashboard created successfully!")
    print("Plots saved to visualizations/ directory")
