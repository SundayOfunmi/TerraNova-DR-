# Project Directory Structure

Below is the layout for the FEMA Disaster Cost Prediction project.

```text
.
├── api/
│   ├── main.py                 # FastAPI application and routes
│   └── schemas.py              # Pydantic data validation models
├── dashboard/
│   └── app.py                  # Streamlit visual interface
├── data/
│   ├── raw/                    # Original CSVs downloaded from FEMA API
│   └── processed/              # Cleaned data with engineered features (NaNs removed)
├── docker/
│   ├── Dockerfile.api          # Build instructions for the Backend
│   └── Dockerfile.dashboard    # Build instructions for the Frontend
├── docs/
│   └── model_card.md           # Model ethics, bias, and performance details
├── models/
│   ├── best_model.pkl          # The trained XGBoost/Random Forest artifact
│   └── shap_summary.png        # Global feature importance visualization
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_feature_analysis.ipynb # Validation of engineered features
│   └── 03_model_evaluation.ipynb # SHAP and residual analysis
├── src/
│   ├── ingestion/
│   │   └── fema_api.py         # Paginated data fetcher with rate limiting
│   ├── processing/
│   │   └── feature_engineering.py # Data cleaning and duration/season engineering
│   ├── models/
│   │   └── train.py            # Training pipeline with MLflow logging
│   └── pipeline/
│   │   └── run_pipeline.py     # Orchestration script (End-to-End)
├── tests/
│   └── test_api_integration.py # Script to verify API endpoint logic
├── .gitignore                  # Prevents data/models from being pushed to GitHub
├── docker-compose.yml          # Multi-container orchestration config
├── requirements.txt            # List of all Python dependencies
└── setup_project.sh            # Automation script to rebuild this structure

