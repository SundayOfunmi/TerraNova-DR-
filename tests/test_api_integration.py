import requests
import json

# Use localhost if running the script on your machine while Docker is up
# If running inside a container, use 'api:8000'
BASE_URL = "http://localhost:8000"

def test_api_health():
    print("Checking API Health...")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        print("✅ Health Check Passed:", response.json())
    else:
        print("❌ Health Check Failed:", response.text)

def test_prediction_endpoint():
    print("\nSending Dummy Disaster Declaration...")
    
    # Matches the 'PredictRequest' schema in your schemas.py
    dummy_payload = {
        "incidentType": "Hurricane",
        "state": "FL",
        "region": 4,
        "declaration_year": 2024,
        "season": "Summer",
        "project_count": 50,
        "avg_project_amount": 125000.0
    }

    try:
        response = requests.post(f"{BASE_URL}/predict-cost", json=dummy_payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction Successful!")
            print(f"   Predicted Cost: ${result['predicted_total_cost_usd']:,.2f}")
            print(f"   Model Version: {result['model_version']}")
        elif response.status_code == 422:
            print("❌ Validation Error (Check your schema):", response.json())
        else:
            print(f"❌ Error {response.status_code}:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Refused. Is the Docker container running on port 8000?")

if __name__ == "__main__":
    test_api_health()
    test_prediction_endpoint()

