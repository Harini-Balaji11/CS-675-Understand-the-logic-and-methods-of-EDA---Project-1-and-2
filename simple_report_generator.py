"""
Simple Report Generator - Windows Compatible
Generate comprehensive reports for stakeholders
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path
import base64
from io import BytesIO

class SimpleReportGenerator:
    """Generate comprehensive reports for stakeholders"""
    
    def __init__(self, df, model_results=None):
        self.df = df
        self.model_results = model_results
        self.report_date = datetime.now().strftime("%Y-%m-%d")
        
    def generate_executive_summary(self):
        """Generate executive summary"""
        total_customers = len(self.df)
        churned_customers = self.df['Churn'].sum()
        churn_rate = (churned_customers / total_customers) * 100
        
        # Revenue metrics
        total_revenue = self.df['MonthlyCharges'].sum()
        churned_revenue = self.df[self.df['Churn'] == 1]['MonthlyCharges'].sum()
        revenue_at_risk = (churned_revenue / total_revenue) * 100
        
        # Key insights
        avg_tenure_churned = self.df[self.df['Churn'] == 1]['tenure'].mean()
        avg_tenure_retained = self.df[self.df['Churn'] == 0]['tenure'].mean()
        
        # Contract analysis
        contract_churn = self.df.groupby('Contract')['Churn'].mean() * 100
        month_to_month_churn = contract_churn.get('Month-to-month', 0)
        
        summary = {
            'report_date': self.report_date,
            'total_customers': total_customers,
            'churn_rate': churn_rate,
            'churned_customers': churned_customers,
            'total_revenue': total_revenue,
            'churned_revenue': churned_revenue,
            'revenue_at_risk_pct': revenue_at_risk,
            'avg_tenure_churned': avg_tenure_churned,
            'avg_tenure_retained': avg_tenure_retained,
            'month_to_month_churn_rate': month_to_month_churn
        }
        
        return summary
    
    def create_visualizations(self):
        """Create key visualizations for the report"""
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Telco Customer Churn Analysis - Key Insights', 
                    fontsize=16, fontweight='bold')
        
        # 1. Overall churn distribution
        churn_counts = self.df['Churn'].value_counts()
        axes[0, 0].pie(churn_counts.values, labels=['Retained', 'Churned'], 
                      autopct='%1.1f%%', startangle=90, colors=['#2E8B57', '#DC143C'])
        axes[0, 0].set_title('Overall Churn Distribution', fontweight='bold')
        
        # 2. Churn by contract type
        contract_churn = self.df.groupby('Contract')['Churn'].mean() * 100
        bars = axes[0, 1].bar(contract_churn.index, contract_churn.values,
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 1].set_title('Churn Rate by Contract Type', fontweight='bold')
        axes[0, 1].set_ylabel('Churn Rate (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, rate in zip(bars, contract_churn.values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. Tenure distribution
        churned_tenure = self.df[self.df['Churn'] == 1]['tenure']
        retained_tenure = self.df[self.df['Churn'] == 0]['tenure']
        
        axes[1, 0].hist([retained_tenure, churned_tenure], bins=20, alpha=0.7,
                       label=['Retained', 'Churned'], color=['#2E8B57', '#DC143C'])
        axes[1, 0].set_title('Tenure Distribution by Churn Status', fontweight='bold')
        axes[1, 0].set_xlabel('Tenure (months)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 4. Monthly charges distribution
        churned_charges = self.df[self.df['Churn'] == 1]['MonthlyCharges']
        retained_charges = self.df[self.df['Churn'] == 0]['MonthlyCharges']
        
        axes[1, 1].hist([retained_charges, churned_charges], bins=20, alpha=0.7,
                       label=['Retained', 'Churned'], color=['#2E8B57', '#DC143C'])
        axes[1, 1].set_title('Monthly Charges Distribution by Churn Status', fontweight='bold')
        axes[1, 1].set_xlabel('Monthly Charges ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path('reports') / f'churn_analysis_{self.report_date}.png'
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def generate_json_report(self):
        """Generate JSON report for API consumption"""
        report_data = {
            'report_metadata': {
                'generated_date': self.report_date,
                'total_customers': len(self.df),
                'report_type': 'churn_analysis'
            },
            'executive_summary': self.generate_executive_summary(),
            'key_metrics': {
                'churn_rate': (self.df['Churn'].sum() / len(self.df)) * 100,
                'avg_tenure': self.df['tenure'].mean(),
                'avg_monthly_charges': self.df['MonthlyCharges'].mean(),
                'total_revenue': self.df['MonthlyCharges'].sum()
            }
        }
        
        # Save JSON report
        report_path = Path('reports') / f'churn_analysis_report_{self.report_date}.json'
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(report_data, indent=2))
        
        return str(report_path)

def generate_simple_reports(df):
    """Generate simple reports"""
    print("Generating comprehensive reports...")
    
    generator = SimpleReportGenerator(df)
    
    # Generate reports
    plot_path = generator.create_visualizations()
    json_report = generator.generate_json_report()
    
    print(f"Visualization saved: {plot_path}")
    print(f"JSON report generated: {json_report}")
    
    return {
        'plot_path': plot_path,
        'json_report': json_report
    }

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/telco-customer-churn.csv')
    
    # Generate reports
    reports = generate_simple_reports(df)
    
    print("\nAll reports generated successfully!")
    print("Check the 'reports/' directory for generated files")
