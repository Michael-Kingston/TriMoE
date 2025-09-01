import pandas as pd
import numpy as np  
from sklearn.preprocessing import StandardScaler

# 1. load
df = pd.read_csv("raw_collected_static_features.csv")
print("[INFO] Raw data loaded.")

# 2. data type conversion
df['Ethnicity'] = df['Ethnicity'].astype(str)
df['Gender'] = df['Gender'].astype(str)
print("[INFO] Corrected initial dtypes for 'Ethnicity' and 'Gender'.")

# 3. feature engineering
# capping age
df['Age'] = df['Age'].apply(lambda x: 90 if x > 89 else x)
print("[INFO] Capped all ages > 89 at 90.")

# height and weight
for col, min_val, max_val in [('Height', 120, 220), ('Weight', 30, 250)]:
    invalid_mask = (df[col] < min_val) | (df[col] > max_val) | (df[col].isnull())
    df[f'{col}_is_missing'] = invalid_mask.astype(int)
    df.loc[invalid_mask, col] = np.nan
    print(f"[INFO] Invalidated and marked {invalid_mask.sum()} '{col}' entries for imputation.")

# language
def clean_language(lang_str):
    if pd.isna(lang_str) or lang_str == '': return 'MISSING'
    if 'ENGL' in str(lang_str).upper(): return 'ENGL'
    return 'OTHER'
df['LANGUAGE'] = df['LANGUAGE'].apply(clean_language)
print("[INFO] Processed 'LANGUAGE' column.")

# marital status
marital_status_map = {'UNKNOWN (DEFAULT)': 'UNKNOWN', 'LIFE PARTNER': 'MARRIED'}
df['MARITAL_STATUS'].fillna('UNKNOWN', inplace=True)
df['MARITAL_STATUS'] = df['MARITAL_STATUS'].replace(marital_status_map)
print("[INFO] Processed 'MARITAL_STATUS' column.")

# admission location
admission_location_map = {'EMERGENCY ROOM ADMIT': 'EMERGENCY', 'CLINIC REFERRAL/PREMATURE': 'CLINIC_REFERRAL',
    'PHYS REFERRAL/NORMAL DELI': 'PHYSICIAN_REFERRAL', 'TRANSFER FROM HOSP/EXTRAM': 'TRANSFER',
    'HMO REFERRAL/SICK': 'HMO_REFERRAL',}

df['ADMISSION_LOCATION'] = df['ADMISSION_LOCATION'].map(admission_location_map)
df['ADMISSION_LOCATION'].fillna('OTHER', inplace=True)
print("[INFO] Processed 'ADMISSION_LOCATION' column.")

# 4. imputation
# for continuous
numerical_cols = ['Age', 'Height', 'Weight']
for col in numerical_cols:
    median_val = df[col].mean() 
    df[col].fillna(median_val, inplace=True)
print("[INFO] Imputed missing numerical values with median.")

# 5. feature processing
categorical_cols = ['Ethnicity', 'Gender', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'MARITAL_STATUS']


# ohe
print(f"\n[INFO] Columns to be one-hot encoded: {categorical_cols}")
processed_df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, prefix_sep='_')
print(f"[INFO] One-hot encoding complete. New shape: {processed_df.shape}")

# scale
scaler = StandardScaler()
processed_df[numerical_cols] = scaler.fit_transform(processed_df[numerical_cols])
print("[INFO] Standardized numerical columns.")

# 6. save
id_cols = ['stay_id'] 

indicator_cols = ['Height_is_missing', 'Weight_is_missing']
ohe_cols = [col for col in processed_df.columns if any(col.startswith(prefix + '_') for prefix in categorical_cols)]

final_cols = id_cols + numerical_cols + indicator_cols + ohe_cols
final_cols = [col for col in final_cols if col in processed_df.columns]  # Sanity check

final_df = processed_df[final_cols]

print(f"\n[INFO] Final processed data shape: {final_df.shape}")
print("--- Final data sample ---")
print(final_df.head())

final_df.to_csv("processed_static_features.csv", index=False)
print("\n✅ Successfully created 'processed_static_features.csv'")