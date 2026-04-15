from pydantic import BaseModel, Field, validator
from typing import Optional

class PredictRequest(BaseModel):
    """
    Input schema for cost prediction. 
    Field names must match the columns used in the training script.
    """
    incidentType: str = Field(..., example="Hurricane", description="The type of disaster incident")
    state: str = Field(..., min_length=2, max_length=2, example="NY", description="Two-letter state abbreviation")
    region: int = Field(..., ge=1, le=10, example=2, description="FEMA Region number (1-10)")
    declaration_year: int = Field(..., ge=1953, example=2024, description="Year of the disaster declaration")
    season: str = Field(..., example="Summer", description="Season of declaration: Winter, Spring, Summer, Autumn")
    project_count: Optional[int] = Field(0, description="Total number of Public Assistance projects")
    avg_project_amount: Optional[float] = Field(0.0, description="Average dollar amount per project")

    @validator('incidentType')
    def validate_incident(cls, v):
        # Optional: Add specific incident validation if desired
        allowed = ['Hurricane', 'Flood', 'Fire', 'Tornado', 'Severe Storm', 'Snow', 'Other']
        # We don't strictly block others to maintain 'handle_unknown' model behavior, 
        # but you could add a warning or normalization here.
        return v.title()

class PredictResponse(BaseModel):
    """
    Output schema for the prediction response.
    """
    predicted_total_cost_usd: float = Field(..., example=1500000.50)
    currency: str = "USD"
    model_version: str = Field(..., example="FEMA_XGB_v1")
    prediction_timestamp: str
    
class HealthResponse(BaseModel):
    """
    Schema for the /health endpoint.
    """
    status: str
    model_loaded: bool
    api_version: str = "1.0.0"

