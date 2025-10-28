"""
Deployment Scripts for Telco Churn Analysis
Automated deployment to various platforms
"""

import os
import subprocess
import json
from pathlib import Path

class DeploymentManager:
    """Manage deployment to various platforms"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.requirements_file = self.project_root / "requirements.txt"
        
    def create_streamlit_config(self):
        """Create Streamlit configuration for deployment"""
        config_content = """
[server]
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
"""
        
        config_file = self.project_root / ".streamlit" / "config.toml"
        config_file.parent.mkdir(exist_ok=True)
        config_file.write_text(config_content)
        print("✅ Streamlit config created")
    
    def create_heroku_files(self):
        """Create files needed for Heroku deployment"""
        # Procfile
        procfile_content = "web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0"
        procfile = self.project_root / "Procfile"
        procfile.write_text(procfile_content)
        
        # runtime.txt
        runtime_content = "python-3.9.18"
        runtime_file = self.project_root / "runtime.txt"
        runtime_file.write_text(runtime_content)
        
        # setup.sh
        setup_content = """#!/bin/bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml"""
        
        setup_file = self.project_root / "setup.sh"
        setup_file.write_text(setup_content)
        setup_file.chmod(0o755)
        
        print("✅ Heroku deployment files created")
    
    def create_docker_files(self):
        """Create Docker configuration"""
        dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data reports visualizations

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
        
        dockerfile = self.project_root / "Dockerfile"
        dockerfile.write_text(dockerfile_content)
        
        # docker-compose.yml
        compose_content = """version: '3.8'

services:
  churn-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
"""
        
        compose_file = self.project_root / "docker-compose.yml"
        compose_file.write_text(compose_content)
        
        # .dockerignore
        dockerignore_content = """__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.DS_Store
.vscode
.idea
*.swp
*.swo
"""
        
        dockerignore = self.project_root / ".dockerignore"
        dockerignore.write_text(dockerignore_content)
        
        print("✅ Docker files created")
    
    def create_github_actions_deploy(self):
        """Create GitHub Actions for automated deployment"""
        workflow_content = """name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v
    
    - name: Train model
      run: |
        python train_model.py
    
    - name: Generate analytics
      run: |
        python advanced_analytics.py
    
    - name: Deploy to Streamlit Cloud
      if: github.ref == 'refs/heads/main'
      uses: streamlit/streamlit-action@main
      with:
        app_file: streamlit_app.py
        cloud_url: ${{ secrets.STREAMLIT_CLOUD_URL }}
        cloud_token: ${{ secrets.STREAMLIT_CLOUD_TOKEN }}
"""
        
        workflow_file = self.project_root / ".github" / "workflows" / "deploy.yml"
        workflow_file.parent.mkdir(parents=True, exist_ok=True)
        workflow_file.write_text(workflow_content)
        
        print("✅ GitHub Actions deployment workflow created")
    
    def create_deployment_script(self):
        """Create deployment script"""
        script_content = """#!/bin/bash

echo "🚀 Starting Telco Churn Analysis Deployment"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found. Please run from project root."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models data reports visualizations .streamlit

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Train model
echo "🤖 Training model..."
python train_model.py

# Generate analytics
echo "📊 Generating analytics..."
python advanced_analytics.py

# Create deployment files
echo "🔧 Creating deployment files..."
python deployment_scripts.py

echo "✅ Deployment preparation complete!"
echo ""
echo "Next steps:"
echo "1. For Streamlit Cloud: Push to GitHub and connect repository"
echo "2. For Heroku: Run 'heroku create' and 'git push heroku main'"
echo "3. For Docker: Run 'docker-compose up'"
echo "4. For local: Run 'streamlit run streamlit_app.py'"
"""
        
        script_file = self.project_root / "deploy.sh"
        script_file.write_text(script_content)
        script_file.chmod(0o755)
        
        print("✅ Deployment script created")
    
    def create_readme_deployment(self):
        """Create deployment README"""
        readme_content = """# 🚀 Deployment Guide

This guide covers various deployment options for the Telco Customer Churn Analysis application.

## 📋 Prerequisites

- Python 3.8+
- Git
- Required Python packages (see requirements.txt)

## 🌐 Deployment Options

### 1. Streamlit Cloud (Recommended)

**Easiest and fastest deployment option**

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add deployment files"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file to `streamlit_app.py`
   - Click "Deploy"

3. **Your app will be live at**: `https://your-app-name.streamlit.app`

### 2. Heroku

**Good for production deployments**

1. **Install Heroku CLI**:
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Deploy**:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

3. **Your app will be live at**: `https://your-app-name.herokuapp.com`

### 3. Docker

**Good for containerized deployments**

1. **Build and run**:
   ```bash
   docker-compose up --build
   ```

2. **Access at**: `http://localhost:8501`

### 4. Local Development

**For testing and development**

1. **Run locally**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access at**: `http://localhost:8501`

## 🔧 Configuration

### Environment Variables

Set these environment variables for production:

```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
```

### Model Training

Before deployment, ensure your model is trained:

```bash
python train_model.py
```

This will create the necessary model files in the `models/` directory.

## 📊 Monitoring

### Health Checks

The application includes health check endpoints:
- `/health` - Basic health check
- `/_stcore/health` - Streamlit health check

### Logs

Monitor application logs:
- **Streamlit Cloud**: Available in the dashboard
- **Heroku**: `heroku logs --tail`
- **Docker**: `docker-compose logs -f`

## 🚨 Troubleshooting

### Common Issues

1. **Model not found**:
   - Ensure `train_model.py` has been run
   - Check that `models/` directory exists

2. **Data file not found**:
   - Ensure `data/telco-customer-churn.csv` exists
   - Check file permissions

3. **Port conflicts**:
   - Change port in configuration
   - Use environment variables

### Performance Optimization

1. **Caching**:
   - Use `@st.cache_data` for expensive operations
   - Cache model loading

2. **Resource Management**:
   - Monitor memory usage
   - Optimize data loading

## 🔒 Security

### Production Security

1. **Environment Variables**:
   - Never commit secrets to Git
   - Use environment variables for sensitive data

2. **Access Control**:
   - Implement authentication if needed
   - Use HTTPS in production

## 📈 Scaling

### Horizontal Scaling

1. **Load Balancing**:
   - Use multiple instances
   - Implement session management

2. **Database**:
   - Use external database for production
   - Implement connection pooling

## 🆘 Support

If you encounter issues:

1. Check the logs
2. Verify all dependencies are installed
3. Ensure model files exist
4. Check file permissions

For additional help, open an issue on GitHub.
"""
        
        readme_file = self.project_root / "DEPLOYMENT.md"
        readme_file.write_text(readme_content)
        
        print("✅ Deployment README created")

def main():
    """Main deployment setup function"""
    print("🚀 Setting up deployment configurations...")
    
    deployer = DeploymentManager()
    
    # Create all deployment files
    deployer.create_streamlit_config()
    deployer.create_heroku_files()
    deployer.create_docker_files()
    deployer.create_github_actions_deploy()
    deployer.create_deployment_script()
    deployer.create_readme_deployment()
    
    print("\n✅ All deployment files created successfully!")
    print("\n📋 Next steps:")
    print("1. Run: chmod +x deploy.sh")
    print("2. Run: ./deploy.sh")
    print("3. Choose your deployment platform")
    print("4. Follow the specific deployment guide")

if __name__ == "__main__":
    main()
