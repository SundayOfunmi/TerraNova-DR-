# FEMA Disaster Cost Prediction & Aggregation Model

An end-to-end Machine Learning ecosystem designed to ingest OpenFEMA data, compute total disaster recovery obligations, and provide real-time cost forecasting via a REST API and Streamlit dashboard.

## 🏗 Architecture Overview
`Ingestion (API)` → `Validation` → `Feature Engineering` → `ML Training (MLflow)` → `FastAPI` → `Streamlit`

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (if running locally)
- [OpenFEMA API Access](https://fema.gov) (No key required for standard rate limits)

### Installation & Deployment
The easiest way to run the full stack (API + Dashboard) is using Docker Compose:

```bash
# Clone the repository
git clone https://github.com
cd fema-cost-prediction

# Build and start services
docker-compose up --build

