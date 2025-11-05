#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Preparation Script for Flight Delay Prediction
====================================================
This script handles all data cleaning and preprocessing steps for the flight 
delay prediction project. It filters data to ATL airport, performs feature 
engineering, and outputs cleaned data ready for model training.

Input: ./data/Combined_Flights_2022.csv
Output: ./data/processed/cleaned_data.csv
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

INPUT_FILE = "./data/Combined_Flights_2022.csv"
OUTPUT_DIR = "./data/processed/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cleaned_data.csv")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("FLIGHT DELAY DATA PREPARATION SCRIPT")
print("="*80)
print(f"\nInput file: {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")

# ==============================================================================
# Step 1: Load Data
# ==============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

df = pd.read_csv(INPUT_FILE)
print(f"\nOriginal dataset shape: {df.shape}")
print(f"Total flights in dataset: {len(df):,}")

# ==============================================================================
# Step 2: Filter to ATL Airport
# ==============================================================================

print("\n" + "="*80)
print("STEP 2: FILTERING TO ATL AIRPORT")
print("="*80)

"""
Due to computational constraints and ATL's status as one of the world's busiest 
airports, we exclusively use data from Hartsfield-Jackson Atlanta International 
Airport. This focused approach ensures efficient use of resources while capturing 
diverse flight scenarios.
"""

specific_origin_airport = ['ATL']  
df_filtered = df[df['Origin'].isin(specific_origin_airport)]
print(f"\nFiltered dataset shape: {df_filtered.shape}")
print(f"Total ATL flights: {len(df_filtered):,}")

# ==============================================================================
# Step 3: Drop ID and Unnecessary Columns
# ==============================================================================

print("\n" + "="*80)
print("STEP 3: DROPPING ID AND UNNECESSARY COLUMNS")
print("="*80)

"""
These columns represent unique IDs and codes which don't provide much value
in training the model and are unrelated to the data analysis.
This would overfit the model and make it less accurate.
"""

id_columns_to_drop = [
    'DOT_ID_Marketing_Airline', 'DOT_ID_Operating_Airline', 'OriginAirportID',
    'OriginAirportSeqID', 'OriginCityMarketID', 'DestAirportID', 'DestAirportSeqID',
    'DestCityMarketID', 'Flight_Number_Marketing_Airline', 'Flight_Number_Operating_Airline',
    'IATA_Code_Marketing_Airline', 'IATA_Code_Operating_Airline', 'Tail_Number',
    'Marketing_Airline_Network', 'Operated_or_Branded_Code_Share_Partners',
    'OriginCityName', 'OriginState', 'DestCityName', 'DestState',
    'OriginStateName', 'OriginStateFips', 'OriginWac',
    'DestStateName', 'DestStateFips', 'DestWac', 'Operating_Airline'
]

df_filtered = df_filtered.drop(columns=id_columns_to_drop)
print(f"\nColumns remaining after dropping IDs: {len(df_filtered.columns)}")

# ==============================================================================
# Step 4: Select Relevant Features
# ==============================================================================

print("\n" + "="*80)
print("STEP 4: SELECTING RELEVANT FEATURES")
print("="*80)

"""
Selected features that are crucial for predicting flight delays:
- TaxiIn, TaxiOut: Taxi-in and taxi-out times
- ArrDel15, DepDel15: Binary indicators of 15+ minute delays
- DayofMonth, Month: Temporal features
- DepTime, WheelsOff, WheelsOn: Key flight event times
- Distance: Flight distance
- CRSDepTime, CRSArrTime: Scheduled times
- Airline: Operating airline
- AirTime: Time spent in the air
- DepHour: Departure hour (computed from CRSDepTime)
"""

# First create DepHour feature from raw data
df_filtered['DepHour'] = df_filtered['CRSDepTime'] // 100

columns_to_keep = [
    'TaxiIn', 'TaxiOut', 'ArrDel15', 'DepDel15', 'DayofMonth', 'Month',
    'DepTime', 'WheelsOff', 'WheelsOn', 'Distance', 'CRSDepTime', 'CRSArrTime',
    'Airline', 'AirTime', 'DepHour'
]

df_filtered = df_filtered[columns_to_keep]
print(f"\nTotal features selected: {len(columns_to_keep)}")
print(f"Features: {', '.join(columns_to_keep)}")

# ==============================================================================
# Step 5: Transform Time Columns (HHMM format)
# ==============================================================================

print("\n" + "="*80)
print("STEP 5: TRANSFORMING TIME FEATURES")
print("="*80)

"""
Time columns are in HHMM format (e.g., 1430 = 2:30 PM). We apply trigonometric 
transformations to convert these into cyclical features using sine and cosine 
functions. This preserves the circular nature of time (e.g., 2359 and 0001 are 
close in time).

For each time column:
1. Extract hours and minutes
2. Convert to minutes since midnight
3. Apply sine and cosine transformations
4. Drop original time columns
"""

time_columns = ['CRSDepTime', 'DepTime', 'CRSArrTime', 'WheelsOff', 'WheelsOn']

for column in time_columns:
    # Extract hour and minute
    df_filtered[column + '_hour'] = df_filtered[column] // 100
    df_filtered[column + '_minute'] = df_filtered[column] % 100
    
    # Apply trigonometric transformations (cyclical encoding)
    minutes_total = df_filtered[column + '_hour'] * 60 + df_filtered[column + '_minute']
    df_filtered[column + '_sin'] = np.sin(2 * np.pi * minutes_total / (24 * 60))
    df_filtered[column + '_cos'] = np.cos(2 * np.pi * minutes_total / (24 * 60))
    
    # Drop the original and intermediate columns
    df_filtered = df_filtered.drop(columns=[column, column + '_hour', column + '_minute'])

print(f"\nTime columns transformed: {', '.join(time_columns)}")
print(f"New features created: {len(time_columns) * 2} (sin/cos pairs)")

# ==============================================================================
# Step 6: Remove Duplicates
# ==============================================================================

print("\n" + "="*80)
print("STEP 6: REMOVING DUPLICATES")
print("="*80)

duplicates_count = df_filtered.duplicated().sum()
print(f"\nDuplicates found: {duplicates_count:,}")

df_filtered = df_filtered.drop_duplicates()
print(f"Duplicates removed. New shape: {df_filtered.shape}")

# ==============================================================================
# Step 7: Handle Missing Values
# ==============================================================================

print("\n" + "="*80)
print("STEP 7: HANDLING MISSING VALUES")
print("="*80)

missing_before = df_filtered.isnull().sum()
total_missing = missing_before.sum()
print(f"\nTotal missing values before preprocessing: {total_missing:,}")
print("\nMissing values by column:")
for col in missing_before[missing_before > 0].index:
    print(f"  {col}: {missing_before[col]:,}")

# ==============================================================================
# Step 8: Apply Preprocessing Pipeline
# ==============================================================================

print("\n" + "="*80)
print("STEP 8: APPLYING PREPROCESSING PIPELINE")
print("="*80)

"""
Preprocessing Pipeline:
1. Numerical features: Impute missing values (mean) → Standardize (zero mean, unit variance)
2. Categorical features: Impute missing values (most frequent) → One-hot encode
3. Target columns (ArrDel15, DepDel15): Pass through without transformation
"""

# Define numerical and categorical columns
num_cols = df_filtered.select_dtypes(include='number').columns.to_list()
cat_cols = df_filtered.select_dtypes(exclude='number').columns.to_list()

# Remove target columns from numerical columns
num_cols.remove('DepDel15')
num_cols.remove('ArrDel15')

print(f"\nNumerical features: {len(num_cols)}")
print(f"Categorical features: {len(cat_cols)}")

# Create pipelines
num_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())

# Combine pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ],
    remainder='passthrough'
)

# Apply preprocessing
df_prepared = preprocessor.fit_transform(df_filtered)

# Get feature names and create DataFrame
feature_names = preprocessor.get_feature_names_out()
df_prepared = pd.DataFrame(data=df_prepared, columns=feature_names)

print(f"\nPreprocessed dataset shape: {df_prepared.shape}")
print(f"Total features after preprocessing: {len(feature_names)}")

# ==============================================================================
# Step 9: Clean Target Variables
# ==============================================================================

print("\n" + "="*80)
print("STEP 9: CLEANING TARGET VARIABLES")
print("="*80)

"""
We drop rows with missing values in target columns (ArrDel15, DepDel15).
We cannot impute these as ArrDel15 is our prediction target, and DepDel15 
could introduce bias if imputed.
"""

rows_before = len(df_prepared)
df_prepared = df_prepared.dropna(subset=['remainder__ArrDel15', 'remainder__DepDel15'])
rows_after = len(df_prepared)
rows_dropped = rows_before - rows_after

print(f"\nRows before cleaning: {rows_before:,}")
print(f"Rows after cleaning: {rows_after:,}")
print(f"Rows dropped: {rows_dropped:,}")

# Convert target columns to integer type
df_prepared['remainder__ArrDel15'] = df_prepared['remainder__ArrDel15'].astype(int)
df_prepared['remainder__DepDel15'] = df_prepared['remainder__DepDel15'].astype(int)

# Check class distribution
target_distribution = df_prepared['remainder__ArrDel15'].value_counts()
print(f"\nTarget variable (ArrDel15) distribution:")
print(f"  No delay (0): {target_distribution[0]:,} ({target_distribution[0]/len(df_prepared)*100:.1f}%)")
print(f"  Delayed (1): {target_distribution[1]:,} ({target_distribution[1]/len(df_prepared)*100:.1f}%)")

# ==============================================================================
# Step 10: Save Cleaned Data
# ==============================================================================

print("\n" + "="*80)
print("STEP 10: SAVING CLEANED DATA")
print("="*80)

# Verify no missing values remain
missing_after = df_prepared.isnull().sum().sum()
print(f"\nTotal missing values after preprocessing: {missing_after}")

# Save to CSV
df_prepared.to_csv(OUTPUT_FILE, index=False)
print(f"\nCleaned data saved to: {os.path.abspath(OUTPUT_FILE)}")
print(f"Final dataset shape: {df_prepared.shape}")
print(f"  - Samples: {df_prepared.shape[0]:,}")
print(f"  - Features: {df_prepared.shape[1]:,}")

# ==============================================================================
# Summary Statistics
# ==============================================================================

print("\n" + "="*80)
print("DATA PREPARATION COMPLETE - SUMMARY")
print("="*80)

print(f"""
Data Preparation Summary:
------------------------
Original dataset:        {len(df):,} flights
Filtered to ATL:         {len(df_filtered):,} flights
After deduplication:     {rows_before:,} flights
Final cleaned dataset:   {rows_after:,} flights
Features after encoding: {df_prepared.shape[1]:,}

Target Distribution:
- No delay (0):  {target_distribution[0]:,} ({target_distribution[0]/len(df_prepared)*100:.1f}%)
- Delayed (1):   {target_distribution[1]:,} ({target_distribution[1]/len(df_prepared)*100:.1f}%)

Output saved to: {OUTPUT_FILE}
""")

print("="*80)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*80)

