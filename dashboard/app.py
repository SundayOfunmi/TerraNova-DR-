import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION ---
# Use 'localhost' if running locally, or 'api' if running inside Docker
API_URL = "http://localhost:8000" 

st.set_page_config(page_title="FEMA Disaster Cost Forecaster", layout="wide")

# --- UI HEADER ---
st.title("📊 Disaster Recovery Cost Prediction Dashboard")
st.markdown("""
This tool forecasts federal recovery obligations based on disaster parameters 
at the point of declaration.
""")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Disaster Parameters")
    
    incident_type = st.selectbox(
        "Incident Type",
        ["Hurricane", "Flood", "Fire", "Tornado", "Severe Storm", "Snow", "Other"]
    )
    
    state = st.selectbox(
        "State", 
        ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", 
         "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
         "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", 
         "VA", "WA", "WV", "WI", "WY", "DC"]
    )
    
    region = st.slider("FEMA Region", 1, 10, 2)
    year = st.number_input("Declaration Year", min_value=1953, max_value=2030, value=datetime.now().year)
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])
    
    st.divider()
    st.subheader("Project Estimates (Optional)")
    project_count = st.number_input("Estimated Project Count", min_value=0, value=10)
    avg_amount = st.number_input("Avg Project Amount ($)", min_value=0.0, value=50000.0)

# --- PREDICTION LOGIC ---
if st.button("🚀 Generate Forecast", use_container_width=True):
    # Construct Payload
    payload = {
        "incidentType": incident_type,
        "state": state,
        "region": region,
        "declaration_year": int(year),
        "season": season,
        "project_count": int(project_count),
        "avg_project_amount": float(avg_amount)
    }

    try:
        with st.spinner("Calling API..."):
            response = requests.post(f"{API_URL}/predict-cost", json=payload)
            response.raise_for_status()
            data = response.json()
            
        # Display Result
        predicted_val = data['predicted_total_cost_usd']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Predicted Total Recovery Cost", 
                value=f"${predicted_val:,.2f}",
                delta="Forecasted Obligation"
            )
            
        with col2:
            st.info(f"**Model Version:** {data['model_version']}\n\n**Timestamp:** {data['prediction_timestamp']}")

        # --- BUDGET GAP SIMULATION ---
        st.divider()
        st.subheader("💰 Budget Gap Analysis")
        allocated = st.number_input("Current Allocated Budget ($)", min_value=0.0, value=predicted_val * 0.8)
        
        gap = predicted_val - allocated
        gap_pct = (gap / predicted_val) * 100 if predicted_val > 0 else 0
        
        fig = go.Figure(go.Bar(
            x=['Allocated Budget', 'Forecasted Need'],
            y=[allocated, predicted_val],
            marker_color=['#636EFA', '#EF553B']
        ))
        fig.update_layout(title="Budget vs. Forecast Comparison", yaxis_title="USD ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        if gap > 0:
            st.error(f"⚠️ **Budget Shortfall Detected:** ${gap:,.2f} ({gap_pct:.1f}% underfunded)")
        else:
            st.success(f"✅ **Sufficient Funding:** Budget exceeds forecast by ${abs(gap):,.2f}")

    except Exception as e:
        st.error(f"Could not connect to API at {API_URL}. Ensure the FastAPI server is running.")
        st.caption(f"Error details: {e}")

# --- EXPLAINABILITY SECTION ---
st.divider()
st.subheader("🔍 Feature Importance & Explainability")
st.write("Based on SHAP analysis of the global model.")

try:
    # Attempt to load the pre-generated SHAP image from Task 7
    st.image("models/shap_summary.png", caption="SHAP Global Feature Importance")
except FileNotFoundError:
    st.warning("SHAP summary image not found in 'models/'. Run the evaluation notebook to generate it.")

