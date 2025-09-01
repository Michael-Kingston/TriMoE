import pandas as pd
import os

# TODO: Set path to your main MIMIC-III CSV directory
MIMIC_DIR = "/home/mikey/mimic-csv"

# Load the ICUSTAYS table
icustays_df = pd.read_csv(os.path.join(MIMIC_DIR, "ICUSTAYS.csv"))

# Create the simple mapping table
mapping_df = icustays_df[['ICUSTAY_ID', 'HADM_ID']]

# Save the mapping file
output_path = "icustay_to_hadm_map.csv"
mapping_df.to_csv(output_path, index=False)

print(f"Mapping file created successfully at: {output_path}")
print(mapping_df.head())