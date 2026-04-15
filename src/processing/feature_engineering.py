import pandas as pd
import numpy as np

def run_feature_engineering():
    # Load
    dec = pd.read_csv("data/raw/declarations.csv")
    pa = pd.read_csv("data/raw/public_assistance.csv")
    web = pd.read_csv("data/raw/disaster_summaries.csv")

    # 1. Aggregate PA obligations to disaster level
    pa_agg = pa.groupby('disasterNumber').agg(
        project_count=('pwNumber', 'count'),
        avg_project_amount=('projectAmount', 'mean'),
        pa_total_obligated=('totalObligated', 'sum')
    ).reset_index()

    # 2. Merge
    df = dec.merge(pa_agg, on='disasterNumber', how='left')
    df = df.merge(web[['disasterNumber', 'totalAmountIhpApproved', 'totalObligatedAmountHmgp']], 
                  on='disasterNumber', how='left')

    # 3. Define Target (Total Obligated)
    df['total_cost'] = df['pa_total_obligated'].fillna(0) + \
                       df['totalAmountIhpApproved'].fillna(0) + \
                       df['totalObligatedAmountHmgp'].fillna(0)
    
    # Remove zeros for log transform
    df = df[df['total_cost'] > 0].copy()
    df['target_log'] = np.log1p(df['total_cost'])

    # 4. Temporal Features
    df['declarationDate'] = pd.to_datetime(df['declarationDate'])
    df['declaration_year'] = df['declarationDate'].dt.year
    df['declaration_month'] = df['declarationDate'].dt.month
    
    # 4-Season Encoding
    df['season'] = df['declaration_month'].map({
        12:'Winter', 1:'Winter', 2:'Winter',
        3:'Spring', 4:'Spring', 5:'Spring',
        6:'Summer', 7:'Summer', 8:'Summer',
        9:'Autumn', 10:'Autumn', 11:'Autumn'
    })

    # 5. Risk Flag
    high_cost_types = ['Hurricane', 'Flood', 'Tornado', 'Severe Storm']
    df['high_cost_incident'] = df['incidentType'].isin(high_cost_types).astype(int)

    df.to_csv("data/processed/processed_disasters.csv", index=False)
    print("Feature engineering complete.")

run_feature_engineering()

