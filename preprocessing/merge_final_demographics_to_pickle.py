import os
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

# --- 1. Configuration ---
PICKLE_INPUT_DIR = r"\Users\mikey\Documents\ERP\Final_Data\ihm"
STATIC_FEATURES_CSV_PATH = "processed_static_features.csv"
PICKLE_OUTPUT_DIR = r"\Users\mikey\Documents\ERP\Final_Data\ihm\final_pickle"

def main():
    print("--- Final Merge: Adding Processed Static Features to Pickle Lists ---")

    # --- 2. Load and Prepare Processed Static Features ---
    print(f"Loading static features from: {STATIC_FEATURES_CSV_PATH}")
    try:
        static_df = pd.read_csv(STATIC_FEATURES_CSV_PATH).set_index('stay_id')
        print(f"Successfully loaded and indexed {len(static_df)} static feature records.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load or index '{STATIC_FEATURES_CSV_PATH}'. Error: {e}")
        return

    os.makedirs(PICKLE_OUTPUT_DIR, exist_ok=True)
    print(f"Updated files will be saved in: {PICKLE_OUTPUT_DIR}")

    # --- 3. Process each data split (train, val, test) ---
    for split_name in ["train", "val", "test"]:
        source_filename = f"{split_name}p2x_data.pkl"
        source_path = os.path.join(PICKLE_INPUT_DIR, source_filename)
        
        print(f"\n--- Processing {source_filename} ---")

        try:
            with open(source_path, 'rb') as f:
                patient_data_list = pickle.load(f)
            print(f"Loaded {len(patient_data_list)} records from {source_filename}.")
        except FileNotFoundError:
            print(f"[WARNING] Source file not found: {source_path}. Skipping.")
            continue

        new_patient_data_list = []
        success_count = 0
        failure_count = 0

        for data_dict in tqdm(patient_data_list, desc=f"Merging {split_name} data"):
            try:
                # Get the original name from the pickle, e.g., "51238_episode1_timeseries.csv"
                original_name = data_dict['name']
                
                # --- THE FIX ---
                # Transform the name to match the 'stay_id' format in the CSV
                lookup_key = original_name.replace('_timeseries.csv', '')
                # --- END OF FIX ---

                # Find the corresponding row of features in the static DataFrame
                static_features_row = static_df.loc[lookup_key]
                
                # Convert the row to a NumPy array
                static_features_array = static_features_row.values.astype(np.float32)

                # Add the new feature vector to the dictionary under the 'dem' key
                data_dict['dem'] = static_features_array
                
                new_patient_data_list.append(data_dict)
                success_count += 1

            except KeyError:
                failure_count += 1
                continue
            except Exception as e:
                print(f"\n[ERROR] An unexpected error occurred for patient '{data_dict.get('name', 'UNKNOWN')}': {e}")
                failure_count += 1
        
        print(f"Successfully merged: {success_count} records.")
        print(f"Skipped (no match found): {failure_count} records.")
        
        if new_patient_data_list:
            output_filename = f"{split_name}p2x_data_with_dem.pkl"
            output_path = os.path.join(PICKLE_OUTPUT_DIR, output_filename)
            with open(output_path, 'wb') as f:
                pickle.dump(new_patient_data_list, f)
            print(f"✅ Successfully saved updated data to '{output_path}'")

    print("\n--- Final Merge Process Complete ---")

if __name__ == "__main__":
    main()