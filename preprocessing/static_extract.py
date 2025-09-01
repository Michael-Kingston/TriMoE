import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# --- 1. Configuration ---
BENCHMARK_DATA_DIR = "/home/mikey/mimic3-benchmarks/data/in-hospital-mortality"
STATIC_DATA_ROOT_DIR = "/home/mikey/mimic3-benchmarks/data/root"
MIMIC_DIR = "/home/mikey/mimic-csv"
MAP_FILE_PATH = "icustay_to_hadm_map.csv"

# --- Helper Function ---
def get_static_data_for_episode(static_file_path, icu_to_hadm_map, admissions_table):
    """
    Reads an episode.csv file, combines its data with data from the ADMISSIONS table,
    and returns a dictionary of all static features.
    """
    try:
        episode_static_df = pd.read_csv(static_file_path)
        if episode_static_df.empty:
            return None

        root_data = episode_static_df.iloc[0]

        # --- THE FIX IS HERE ---
        # Check for the correct column name: 'Icustay'
        if 'Icustay' not in root_data:
            # This is a fallback check, just in case.
            if 'icustay_id' not in root_data and 'ICUSTAY_ID' not in root_data:
                # print(f"\n[WARNING] No ICU Stay ID column found in {os.path.basename(static_file_path)}. Skipping.")
                return None
            id_col_to_check = 'icustay_id' if 'icustay_id' in root_data else 'ICUSTAY_ID'
        else:
            id_col_to_check = 'Icustay'

        icustay_id = int(root_data[id_col_to_check])
        # --- END OF FIX ---

        # The rest of the logic remains the same
        if icustay_id not in icu_to_hadm_map.index:
            return None
        hadm_id = int(icu_to_hadm_map.loc[icustay_id, 'HADM_ID'])
        if hadm_id not in admissions_table.index:
            return None

        admission_data = admissions_table.loc[hadm_id]
        combined_data = {
            'icustay_id': icustay_id, 'hadm_id': hadm_id, 'Age': root_data.get('Age'),
            'Ethnicity': root_data.get('Ethnicity'), 'Gender': root_data.get('Gender'),
            'Height': root_data.get('Height'), 'Weight': root_data.get('Weight'),
            'ADMISSION_TYPE': admission_data.get('ADMISSION_TYPE'),
            'ADMISSION_LOCATION': admission_data.get('ADMISSION_LOCATION'),
            'INSURANCE': admission_data.get('INSURANCE'),
            'LANGUAGE': admission_data.get('LANGUAGE'),
            'MARITAL_STATUS': admission_data.get('MARITAL_STATUS')
        }
        return combined_data
    except Exception:
        return None

def main():
    print("--- Building Master Static Feature Table ---")

    try:
        icu_to_hadm_map = pd.read_csv(MAP_FILE_PATH).set_index('ICUSTAY_ID')
        admissions_table = pd.read_csv(os.path.join(MIMIC_DIR, "ADMISSIONS.csv")).set_index('HADM_ID')
    except FileNotFoundError as e:
        print(f"[FATAL] Could not load a required file: {e}")
        return

    all_static_data = []

    for listfile_type in ["train", "val", "test"]:
        listfile_name = f"{listfile_type}_listfile.csv"
        path = os.path.join(BENCHMARK_DATA_DIR, listfile_name)
        try:
            df = pd.read_csv(path)
            print(f"[INFO] Processing {len(df)} episodes from {listfile_name}...")
        except FileNotFoundError:
            print(f"[WARNING] Missing listfile: {path}. Skipping.")
            continue

        df['split'] = 'train' if listfile_type != 'test' else 'test'

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Gathering {listfile_type} data"):
            stay_filename = row['stay']
            if os.path.sep in stay_filename:
                split_from_path, fname = stay_filename.split(os.path.sep, 1)
            else:
                split_from_path = row['split']
                fname = stay_filename

            subject_id = fname.split('_')[0]
            episode_info = fname.split('_')[1]
            static_file_path = os.path.join(STATIC_DATA_ROOT_DIR, split_from_path, subject_id, f"{episode_info}.csv")

            if os.path.exists(static_file_path):
                data_dict = get_static_data_for_episode(static_file_path, icu_to_hadm_map, admissions_table)
                if data_dict:
                    data_dict['stay_id'] = fname.replace('_timeseries.csv', '')
                    all_static_data.append(data_dict)

    print(f"\n✅ Finished gathering. Collected {len(all_static_data)} valid rows.")
    if not all_static_data:
        print("⚠️ WARNING: No data was collected! Check other potential issues.")
        return

    static_df = pd.DataFrame(all_static_data)

    # --- This is where you will add your final processing ---
    # (Cleaning, one-hot encoding, standardization, etc.)
    # For now, let's just save the raw collected data to verify it's working.

    output_path = "raw_collected_static_features.csv"
    static_df.to_csv(output_path, index=False)
    print(f"✅ Raw collected static feature table saved as '{output_path}'")
    print("--- Sample of collected data ---")
    print(static_df.head())
    print(f"DataFrame shape: {static_df.shape}")

if __name__ == "__main__":
    main()