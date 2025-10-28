"""
Automated Report Generation
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

class ReportGenerator:
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
    
    def generate_business_insights(self):
        """Generate business insights and recommendations"""
        insights = []
        
        # Churn rate analysis
        churn_rate = (self.df['Churn'].sum() / len(self.df)) * 100
        if churn_rate > 25:
            insights.append({
                'category': 'Critical',
                'insight': f'High churn rate of {churn_rate:.1f}% requires immediate attention',
                'recommendation': 'Implement urgent retention strategies'
            })
        
        # Contract analysis
        contract_churn = self.df.groupby('Contract')['Churn'].mean() * 100
        month_to_month_churn = contract_churn.get('Month-to-month', 0)
        if month_to_month_churn > 40:
            insights.append({
                'category': 'High Priority',
                'insight': f'Month-to-month contracts have {month_to_month_churn:.1f}% churn rate',
                'recommendation': 'Focus on converting to longer-term contracts'
            })
        
        # Tenure analysis
        avg_tenure_churned = self.df[self.df['Churn'] == 1]['tenure'].mean()
        if avg_tenure_churned < 20:
            insights.append({
                'category': 'Medium Priority',
                'insight': f'Churned customers average {avg_tenure_churned:.1f} months tenure',
                'recommendation': 'Implement early intervention programs'
            })
        
        # Revenue impact
        churned_revenue = self.df[self.df['Churn'] == 1]['MonthlyCharges'].sum()
        total_revenue = self.df['MonthlyCharges'].sum()
        revenue_at_risk = (churned_revenue / total_revenue) * 100
        
        if revenue_at_risk > 20:
            insights.append({
                'category': 'Financial Impact',
                'insight': f'${churned_revenue:,.2f} in revenue at risk ({revenue_at_risk:.1f}%)',
                'recommendation': 'Prioritize high-value customer retention'
            })
        
        return insights
    
    def generate_model_performance_report(self):
        """Generate model performance report"""
        if not self.model_results:
            return None
        
        # Find best model
        best_model = max(self.model_results.keys(), 
                        key=lambda x: self.model_results[x]['test_f1'])
        
        performance_report = {
            'best_model': best_model,
            'model_performance': self.model_results[best_model],
            'all_models': self.model_results,
            'model_comparison': {
                'accuracy_ranking': sorted(self.model_results.keys(), 
                                         key=lambda x: self.model_results[x]['test_accuracy'], 
                                         reverse=True),
                'f1_ranking': sorted(self.model_results.keys(), 
                                  key=lambda x: self.model_results[x]['test_f1'], 
                                  reverse=True)
            }
        }
        
        return performance_report
    
    def create_visualizations(self):
        """Create key visualizations for the report"""
        # Set style
        plt.style.use('seaborn-v0_8')
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
    
    def generate_html_report(self):
        """Generate HTML report"""
        # Generate all components
        executive_summary = self.generate_executive_summary()
        business_insights = self.generate_business_insights()
        model_performance = self.generate_model_performance_report()
        plot_path = self.create_visualizations()
        
        # Convert plot to base64 for HTML embedding
        with open(plot_path, 'rb') as f:
            plot_data = base64.b64encode(f.read()).decode()
        
        # HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Telco Customer Churn Analysis Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    border-bottom: 3px solid #1f77b4;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    color: #1f77b4;
                    margin: 0;
                    font-size: 2.5em;
                }}
                .header p {{
                    color: #666;
                    margin: 10px 0 0 0;
                    font-size: 1.1em;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                .section h2 {{
                    color: #333;
                    border-left: 4px solid #1f77b4;
                    padding-left: 15px;
                    margin-bottom: 20px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .metric-card h3 {{
                    margin: 0 0 10px 0;
                    font-size: 2em;
                }}
                .metric-card p {{
                    margin: 0;
                    opacity: 0.9;
                }}
                .insights {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 4px solid #28a745;
                }}
                .insight-item {{
                    margin-bottom: 15px;
                    padding: 15px;
                    background-color: white;
                    border-radius: 5px;
                    border-left: 3px solid #007bff;
                }}
                .insight-category {{
                    font-weight: bold;
                    color: #007bff;
                    margin-bottom: 5px;
                }}
                .recommendation {{
                    font-style: italic;
                    color: #28a745;
                }}
                .plot-container {{
                    text-align: center;
                    margin: 30px 0;
                }}
                .plot-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Telco Customer Churn Analysis</h1>
                    <p>Comprehensive Analysis Report - {executive_summary['report_date']}</p>
                </div>
                
                <div class="section">
                    <h2>📈 Executive Summary</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <h3>{executive_summary['total_customers']:,}</h3>
                            <p>Total Customers</p>
                        </div>
                        <div class="metric-card">
                            <h3>{executive_summary['churn_rate']:.1f}%</h3>
                            <p>Churn Rate</p>
                        </div>
                        <div class="metric-card">
                            <h3>${executive_summary['churned_revenue']:,.0f}</h3>
                            <p>Revenue at Risk</p>
                        </div>
                        <div class="metric-card">
                            <h3>{executive_summary['month_to_month_churn_rate']:.1f}%</h3>
                            <p>Month-to-Month Churn</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔍 Key Insights & Recommendations</h2>
                    <div class="insights">
        """
        
        # Add insights
        for insight in business_insights:
            html_content += f"""
                        <div class="insight-item">
                            <div class="insight-category">{insight['category']}</div>
                            <div>{insight['insight']}</div>
                            <div class="recommendation">💡 {insight['recommendation']}</div>
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Visual Analysis</h2>
                    <div class="plot-container">
                        <img src="data:image/png;base64,"""
        
        html_content += plot_data
        html_content += """" alt="Churn Analysis Visualizations">
                    </div>
                </div>
        """
        
        # Add model performance if available
        if model_performance:
            html_content += f"""
                <div class="section">
                    <h2>🤖 Model Performance</h2>
                    <div class="insights">
                        <div class="insight-item">
                            <div class="insight-category">Best Model</div>
                            <div>{model_performance['best_model']}</div>
                            <div class="recommendation">F1 Score: {model_performance['model_performance']['test_f1']:.3f}</div>
                        </div>
                    </div>
                </div>
            """
        
        html_content += f"""
                <div class="footer">
                    <p>Report generated on {executive_summary['report_date']}</p>
                    <p>Telco Customer Churn Analysis - Data Science Project</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = Path('reports') / f'churn_analysis_report_{self.report_date}.html'
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(html_content)
        
        return str(report_path)
    
    def generate_json_report(self):
        """Generate JSON report for API consumption"""
        report_data = {
            'report_metadata': {
                'generated_date': self.report_date,
                'total_customers': len(self.df),
                'report_type': 'churn_analysis'
            },
            'executive_summary': self.generate_executive_summary(),
            'business_insights': self.generate_business_insights(),
            'model_performance': self.generate_model_performance_report(),
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

def generate_all_reports(df, model_results=None):
    """Generate all types of reports"""
    print("📊 Generating comprehensive reports...")
    
    generator = ReportGenerator(df, model_results)
    
    # Generate all reports
    html_report = generator.generate_html_report()
    json_report = generator.generate_json_report()
    
    print(f"✅ HTML report generated: {html_report}")
    print(f"✅ JSON report generated: {json_report}")
    
    return {
        'html_report': html_report,
        'json_report': json_report
    }

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/telco-customer-churn.csv')
    
    # Generate reports
    reports = generate_all_reports(df)
    
    print("\n🎉 All reports generated successfully!")
    print("📁 Check the 'reports/' directory for generated files")
