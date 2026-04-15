import os
import joblib
import pandas as pd
from pathlib import Path
from src.ingestion.fema_api import FEMADataIngestor
from src.processing.feature_engineering import run_feature_engineering
from src.models.train import train_with_mlflow
from sklearn.metrics import r2_score

class FemaPipeline:
    def __init__(self, model_path="models/best_model.pkl"):
        self.model_path = Path(model_path)
        self.processed_data = Path("data/processed/processed_disasters.csv")

    def evaluate_current_model(self, X, y):
        """Calculates R2 of the model currently in production."""
        if not self.model_path.exists():
            return -float('inf')
        model = joblib.load(self.model_path)
        preds = model.predict(X)
        return r2_score(y, preds)

    def run(self):
        print("--- Starting Pipeline Automation ---")

        # 1. Ingestion
        print("\n[Step 1/4] Ingesting Data...")
        #ingestor = FEMADataIngestor()
        #ingestor.ingest()

        # 2. Feature Engineering
        print("\n[Step 2/4] Engineering Features...")
        run_feature_engineering()

        # 3. Load processed data for comparison
        df = pd.read_csv(self.processed_data)
        # Assuming features defined in train.py
        features = ['declaration_year', 'region', 'project_count', 'avg_project_amount', 
                    'incidentType', 'state', 'season']
        X = df[features]
        y = df['target_log']

        # 4. Evaluation and Training
        print("\n[Step 3/4] Evaluating current model vs retraining...")
        current_r2 = self.evaluate_current_model(X, y)
        print(f"Current Model R2: {current_r2:.4f}")

        # Retrain (This saves to a temp location or logs to MLflow)
        # Note: In a real scenario, you'd save the 'new' model to a temp file first
        train_with_mlflow() 
        
        # 5. Logic to swap model (Simplified)
        # In Task 12, you should compare new_r2 vs current_r2 before replacing
        print("\n[Step 4/4] Pipeline Complete. Model artifact updated in 'models/'.")

if __name__ == "__main__":
    pipeline = FemaPipeline()
    pipeline.run()

