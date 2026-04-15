import pandas as pd
import requests
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

BASE_URL = "https://fema.gov"
ENDPOINTS = {
    "declarations": "v2/DisasterDeclarationsSummaries",
    "public_assistance": "v2/PublicAssistanceFundedProjectsDetails",
    "disaster_summaries": "v1/FemaWebDisasterSummaries"
}

class FEMADataIngestor:
    def __init__(self, raw_dir="data/raw"):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_paginated(self, endpoint, label):
        url = f"{BASE_URL}/{endpoint}"
        all_data = []
        skip = 0
        top = 1000
        
        print(f"Fetching {label}...")
        while True:
            params = {"$inlinecount": "allpages", "$skip": skip, "$top": top}
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json().get(label, [])
                
                if not data:
                    break
                    
                all_data.extend(data)
                print(f"Downloaded {len(all_data)} records...")
                
                if len(data) < top:
                    break
                    
                skip += top
                time.sleep(0.3) # Rate limiting
            except Exception as e:
                print(f"Error fetching {label}: {e}")
                break
        return pd.DataFrame(all_data)

    def ingest(self):
        for label, endpoint in ENDPOINTS.items():
            file_path = self.raw_dir / f"{label}.csv"
            
            # Freshness Check (7 days)
            if file_path.exists():
                file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age < timedelta(days=7):
                    print(f"Skipping {label}, data is fresh ({file_age.days} days old).")
                    continue

            df = self._fetch_paginated(endpoint, endpoint.split('/')[-1])
            df.to_csv(file_path, index=False)
            print(f"Saved {len(df)} records to {file_path}")

if __name__ == "__main__":
    ingestor = FEMADataIngestor()
    ingestor.ingest()

