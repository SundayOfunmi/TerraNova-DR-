#==========================================================
# Task 5: Feature Analysis
#==========================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/processed_disasters.csv")

# 1. Boxplot of target by categorical features
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(data=df, x='region', y='target_log', ax=axes[0])
sns.boxplot(data=df, x='season', y='target_log', ax=axes[1])
sns.boxplot(data=df, x='high_cost_incident', y='target_log', ax=axes[2])
plt.show()

# 2. Scatter plot: Duration vs Cost
if 'incident_duration_days' in df.columns:
    sns.scatterplot(data=df, x='incident_duration_days', y='target_log', hue='incidentType')
    plt.title("Duration vs Log Cost")
    plt.show()

# 3. Confirm high_cost_incident balance
print("Class Balance (High Cost Flag):")
print(df['high_cost_incident'].value_counts(normalize=True))

