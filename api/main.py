import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from api.schemas import PredictRequest, PredictResponse, HealthResponse

# Initialize FastAPI app
app = FastAPI(
    title="FEMA Disaster Cost Prediction API",
    description="REST API to forecast federal disaster recovery obligations.",
    version="1.0.0"
)

# Global variable to hold the model
model = None
MODEL_PATH = "models/best_model.pkl"

@app.on_event("startup")
def load_model():
    """
    Loads the trained model pipeline on API startup.
    """
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Returns the status of the API and model loading.
    """
    return {
        "status": "online" if model else "degraded",
        "model_loaded": model is not None,
        "api_version": "1.0.0"
    }

@app.post("/predict-cost", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Accepts disaster parameters and returns a cost prediction in USD.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    try:
        # 1. Convert Pydantic request to DataFrame for the sklearn pipeline
        input_dict = request.dict()
        input_df = pd.DataFrame([input_dict])

        # 2. Generate Prediction (Note: Model predicts in log scale)
        log_prediction = model.predict(input_df)[0]

        # 3. Inverse Log Transform (exp(x) - 1) to get original USD value
        usd_prediction = np.expm1(log_prediction)

        # 4. Prepare Response
        return {
            "predicted_total_cost_usd": round(float(usd_prediction), 2),
            "currency": "USD",
            "model_version": "FEMA_XGB_v1",
            "prediction_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

@app.get("/model-info")
def model_info():
    """
    Returns metadata about the deployed model artifact.
    """
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "model_type": str(type(model.named_steps['model'])),
        "features_used": list(model.named_steps['prep'].get_feature_names_out()),
        "last_modified": datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

