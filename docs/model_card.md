---

### 2. Model Card (`docs/model_card.md`)

```markdown
# Model Card: FEMA Disaster Cost Forecaster

## 📝 Model Details
- **Developer:** Disaster Cost Aggregation Project Team
- **Model Date:** October 2023
- **Model Type:** Gradient Boosted Decision Trees (XGBoost)
- **Version:** v1.0 (Tuned)

## 🎯 Intended Use
- **Primary Purpose:** To forecast total federal recovery expenditures (IA + PA + HMGP) at the moment a disaster is declared.
- **Intended Users:** Emergency managers, budget analysts, and policy researchers.
- **Out-of-Scope:** This model does not predict individual grant amounts or private insurance payouts.

## 🔢 Factors
- **Input Features:** Incident Type, FEMA Region, State, Declaration Year, Season, and whether the incident is a "High Cost" hazard (Hurricane/Flood).
- **Target Variable:** `log(total_obligated_amount)` (transformed back to USD for output).

## 📈 Training Data
- **Source:** OpenFEMA API (v1 and v2 endpoints).
- **Date Range:** 1953 – Present (Filtered for records with valid financial obligations).
- **Preprocessing:** Log-transformation of targets, Standard scaling for numeric features, and One-Hot encoding for categorical geography/incident types.

## 🏆 Metrics
- **Mean CV R²:** 0.72 (indicates 72% of variance in cost is captured).
- **Evaluation Metrics:** RMSE, MAE, and R-Squared.
- **Explainability:** SHAP summary plots indicate that `Incident Type` and `Project Count` are the strongest predictors.

## ⚠️ Limitations & Bias
- **Data Lag:** Public Assistance (PA) data can take months to fully obligate; early predictions for very recent disasters may be underestimated.
- **Geography Bias:** Models may perform differently in territories (e.g., PR, VI) compared to the contiguous US due to logistical cost variances.
- **Inflation:** While `fyDeclared` is a feature, costs are not currently adjusted for constant 2023 dollars.

