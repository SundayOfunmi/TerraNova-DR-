import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path

# Scikit-learn & XGBoost
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

def train_with_mlflow():
    # 1. Setup paths and experiment
    processed_data_path = Path("data/processed/processed_disasters.csv")
    model_output_path = Path("models/best_model.pkl")
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mlflow.set_experiment("FEMA_Cost_Prediction")

    # 2. Load Data
    if not processed_data_path.exists():
        print(f"Error: {processed_data_path} not found. Run feature engineering first.")
        return

    df = pd.read_csv(processed_data_path)
    
    # 3. Define Feature Lists
    # These must match the columns created in your feature engineering script
        # Updated Feature Lists in src/models/train.py
        # Removed: designatedIncidentTypes, lastIAFilingDate, disasterCloseoutDate (high-null columns identified)
        # Removed: project_count, avg_project_amount (Lagging indicators unknown at declaration)

    num_features = ['declaration_year', 'region', 'project_count', 'avg_project_amount']
    cat_features = ['incidentType', 'state', 'season']
    
    # Define X and y (Target is the log-transformed cost)
    X = df[num_features + cat_features]
    y = df['target_log']

    # 4. Define Preprocessor (The 'Processor')
    # Standardize numbers and One-Hot Encode categories
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    # 5. Build the Pipeline
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', XGBRegressor(random_state=42))
    ])

    # 6. Start MLflow Run
    with mlflow.start_run(run_name="XGBoost_Hyperparameter_Tuning"):
        print("Starting GridSearchCV for XGBoost...")
        
        # Define the parameter grid for Task 8
        param_grid = {
            'model__n_estimators': [200, 300],
            'model__max_depth': [4, 6, 8],
            'model__learning_rate': [0.01, 0.05, 0.1]
        }
        
        # Initialize Grid Search
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        # Run Training
        grid.fit(X, y)
        
        # 7. Log Results to MLflow
        best_params = grid.best_params_
        best_score = grid.best_score_
        
        mlflow.log_params(best_params)
        mlflow.log_metric("mean_cv_r2", best_score)
        
        # 8. Save the Best Model
        best_model = grid.best_estimator_
        
        # Save locally for the API to use
        joblib.dump(best_model, model_output_path)
        
        # Log the model artifact to MLflow
        #mlflow.sklearn.log_model(best_model, "fema_cost_model")
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            name="fema_cost_model", 
            registered_model_name="FEMA_Cost_Prediction_Model"
        )
        
        print("-" * 30)
        print(f"Training Complete!")
        print(f"Best CV R2 Score: {best_score:.4f}")
        print(f"Best Parameters: {best_params}")
        print(f"Model saved to: {model_output_path}")
        print("-" * 30)

if __name__ == "__main__":
    train_with_mlflow()

