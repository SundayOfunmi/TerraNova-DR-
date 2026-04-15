#==========================================================
# Task 7: Evaluation & Explainability 
#==========================================================
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load model and data
model_pipe = joblib.load("models/best_model.pkl")
df = pd.read_csv("data/processed/processed_disasters.csv")
X = df[['declaration_year', 'region', 'project_count', 'avg_project_amount', 'incidentType', 'state', 'season']]

# Transform data for SHAP (SHAP often needs the array output from the preprocessor)
X_processed = model_pipe.named_steps['prep'].transform(X)
# Get feature names from the OneHotEncoder
cat_names = model_pipe.named_steps['prep'].transformers_[1][1].get_feature_names_out()
all_features = ['declaration_year', 'region', 'project_count', 'avg_project_amount'] + list(cat_names)

# SHAP Analysis
explainer = shap.Explainer(model_pipe.named_steps['model'])
shap_values = explainer(X_processed)

# Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_processed, feature_names=all_features, show=False)
plt.savefig("models/shap_summary.png")
plt.show()

