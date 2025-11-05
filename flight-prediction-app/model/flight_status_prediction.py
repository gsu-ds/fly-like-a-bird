#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory for figures
OUTPUT_DIR = './output_figures/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output figures will be saved to: {os.path.abspath(OUTPUT_DIR)}")

# ==============================================================================
# 1. Problem Statement
# ==============================================================================

"""
Can we predict if the flight is arriving late to the destination?

Dataset Overview:

The dataset used in this analysis contains information on thousands of flights, 
including details such as departure delays, arrival delays, airlines, airports, 
and more. We explore the dataset, preprocess the data, engineer relevant features, 
and train several machine learning models to predict flight delays.

Note: Selection of ATL for Model Training

Due to computational constraints and ATL's status as one of the world's busiest 
airports, our study exclusively trains models on data from Hartsfield-Jackson 
Atlanta International Airport. This focused approach ensures efficient use of 
resources while capturing diverse flight scenarios, enhancing the accuracy of 
our predictive models for flight delays and arrivals.
"""

# ==============================================================================
# 2. Get the data
# ==============================================================================

import zipfile
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO

csv_file_path = "./data/Combined_Flights_2022.csv"

# Read the CSV file using pandas
df = pd.read_csv(csv_file_path)
print("\n=== First 10 rows of the dataset ===")
print(df.head(10))

# ==============================================================================
# 3. Explore and visualize the data to gain insights
# ==============================================================================

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# ### Getting Information about the Dataset

# Getting shape of the dataset
print(f"\n=== Dataset Shape ===")
print(f"Shape: {df.shape}")

# Getting info about the dataset
print("\n=== Dataset Info ===")
df.info()

# Getting summary statistics of the data
print("\n=== Summary Statistics ===")
print(df.describe())

# Extracting column names
columns = df.columns
print(f"\n=== Total Columns: {len(columns)} ===")
print(columns)

"""
Origin is one of the features and the different cities in which the airport 
is located, these are our samples.
"""

# Getting the airport codes 
origin_airtport_city = df['Origin'].unique()
print(f"\n=== Unique Origin Airports: {len(origin_airtport_city)} ===")
print(origin_airtport_city)

"""
Exploring different airports

- Busiest airports
- Our dataset is very large, so we will be considering only the busiest 
  airports for model training
"""

# ### TOP 10 Busiest Airports in USA

# Getting the top 10 most busiest airports in USA for 2022
busiest_airports = df['Origin'].value_counts().nlargest(10)
print("\n=== Top 10 Busiest Airports ===")
print(busiest_airports)

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=busiest_airports.index, y=busiest_airports.values, palette='viridis', hue=busiest_airports.index, legend=False)
plt.title('Top 10 Busiest Airports in USA (2022)')
plt.xlabel('Airport')
plt.ylabel('Number of Flights')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Save figure
figure_path = os.path.join(OUTPUT_DIR, 'top10_busiest_airports.png')
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"Figure saved: {figure_path}")
plt.close()

"""
Observations from Top 10 Busiest Airports:

* ATL (Atlanta) leads as the busiest airport in the USA, indicating a major hub 
  for air traffic.

* ORD (Chicago O'Hare), DFW (Dallas/Fort Worth), and DEN (Denver) follow closely, 
  underlining their importance as central nodes in the US air travel network.

* The gradual decrease in flight numbers from CLT (Charlotte) to PHX (Phoenix) 
  suggests a tiered structure of airport busyness, with a significant drop-off 
  after the top few.

* Airports like LAX (Los Angeles) and SEA (Seattle) maintain a high volume of 
  traffic, emphasizing their roles as significant gateways, especially for 
  international and transpacific travel.

* LGA (LaGuardia) and LAS (Las Vegas), while among the top 10, handle notably 
  fewer flights than the leading airports, reflecting their regional prominence 
  versus national centrality.
"""

# ## Creating a new column DepHour which enables to get insights about average flight delays on hourly basis

# Convert CRSDepTime to hours
df['DepHour'] = df['CRSDepTime'] // 100

# Calculate average delay by hour
average_delays_by_hour = df.groupby('DepHour')['DepDelay'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=average_delays_by_hour, x='DepHour', y='DepDelay')
plt.title('Average Flight Delays by Time of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Delay (minutes)')
plt.xticks(rotation=45)
plt.tight_layout()
# Save figure
figure_path = os.path.join(OUTPUT_DIR, 'average_delays_by_hour.png')
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"Figure saved: {figure_path}")
plt.close()

"""
From the above bar chart graph it is evident that the flights tend to experience 
higher average delays later in the day, with delays gradually increasing from the 
early afternoon hours and peaking in the evening. Early morning flights show lower 
average delays, suggesting that flying earlier might reduce the likelihood of 
encountering delays.
"""

# **Creating correlation matrix on selected numerical columns to get better idea about how data is associated**

numerical_features = df[['ArrDel15','DepDel15', 'WheelsOff' , 'WheelsOn','ArrDelay','DistanceGroup','CRSDepTime', 'CRSArrTime', 'AirTime', 'Distance','DepHour', 'TaxiIn','TaxiOut',]]
correlation_matrix = numerical_features.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
# Save figure
figure_path = os.path.join(OUTPUT_DIR, 'correlation_matrix.png')
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"Figure saved: {figure_path}")
plt.close()

"""
Correlation Matrix Observations:

* Arrival and departure delays (ArrDel15 and DepDel15) have a strong positive 
  correlation, indicating that when a flight departs late, it is also likely to 
  arrive late.

* There is a significant positive correlation between CRSDepTime and DepHour, 
  as well as CRSArrTime and WheelsOn, which is expected since these are directly 
  related to the scheduled times and actual times of flights.
"""

print("\n" + "="*80)
print("------------------------ EDA Ends Here --------------------------")
print("="*80)

# ==============================================================================
# 4. Prepare the data for Machine Learning algorithms
# ==============================================================================

print("\n" + "="*80)
print("DATA PREPARATION")
print("="*80)

"""
Important Note:

Due to the lack of computing power available, since we couldn't run any models 
with all of the dataset and had to create a subset of the dataset where we used 
the busiest airport to train the data set.
"""

# Choosing samples with Origin airport code 'ATL'
specific_origin_airport = ['ATL']  
df_filtered = df[df['Origin'].isin(specific_origin_airport)]
print("\n=== Filtered Dataset (ATL only) - First 5 rows ===")
print(df_filtered.head())

# ### Dropping Unwanted columns with IDs

"""
These columns represent unique IDs and code which don't provide much value
in the training the model and are unrelated to the data analysis.
This would overfit the model and make the model less accurate.
"""

# List of columns to drop (IDs)
id_columns_to_drop = ['DOT_ID_Marketing_Airline', 'DOT_ID_Operating_Airline', 'OriginAirportID',
                      'OriginAirportSeqID', 'OriginCityMarketID', 'DestAirportID', 'DestAirportSeqID',
                      'DestCityMarketID', 'Flight_Number_Marketing_Airline', 'Flight_Number_Operating_Airline',
                      'IATA_Code_Marketing_Airline', 'IATA_Code_Operating_Airline', 'Tail_Number',
                      'Marketing_Airline_Network', 'Operated_or_Branded_Code_Share_Partners',
                   'OriginCityName', 'OriginState', 'DestCityName', 'DestState',
                      'OriginStateName','OriginStateFips','OriginWac'
                  ,'DestStateName','DestStateFips','DestWac','Operating_Airline']

# Drop ID columns from the DataFrame
df_filtered = df_filtered.drop(columns=id_columns_to_drop)

# Print data types of the remaining columns
print("\n=== Data Types After Dropping ID Columns ===")
print(df_filtered.dtypes)

# ### Exploring the rows and columns

# Getting total columns after dropping columns
columns = df_filtered.columns
print(f"\n=== Total Columns After Dropping: {len(columns)} ===")

# Getting total row after filtering
print(f"=== Total Rows After Filtering: {len(df_filtered.index)} ===")

# Getting the count of unique Airlines that take off and arrive to Atlanta airport
airlines = df_filtered['Airline'].unique()
print(f"=== Unique Airlines at ATL: {len(airlines)} ===")

# Getting unique destination
dest = df_filtered['Dest'].unique()
print(f"=== Unique Destinations from ATL: {len(dest)} ===")

# ### Selected Columns/Features for Modeling

"""
These columns are crucial for predicting flight delays and understanding the 
factors influencing arrival delays.

Furthermore, the selected columns include:

- TaxiIn and TaxiOut: These columns represent taxi-in and taxi-out times, which 
  contribute to overall flight duration.
- ArrDel15 and DepDel15: Binary indicators of whether the arrival or departure 
  was delayed by 15 minutes or more.
- DayofMonth and Month: Date-related features providing temporal context for 
  flight delays.
- DepTime, WheelsOff, and WheelsOn: Times indicating departure, wheels-off, and 
  wheels-on, respectively, capturing key events during a flight.
- Distance: The distance traveled by the flight, which can impact the likelihood 
  of delays.
- CRSDepTime and CRSArrTime: Scheduled departure and arrival times, serving as 
  reference points for delay calculations.
- Airline: The airline operating the flight, potentially influencing delay patterns.
- AirTime: Actual time spent in the air during the flight.
- DepHour: Gives information about average flight delays on an hourly basis
"""

columns_to_keep = [
    'TaxiIn', 'TaxiOut','ArrDel15','DepDel15','DayofMonth','Month',
    'DepTime' ,'WheelsOff', 'WheelsOn', 'Distance','CRSDepTime','CRSArrTime',
     'Airline','AirTime', 'DepHour'
]

df_filtered = df_filtered[columns_to_keep]

"""
We dropped other columns because they will have too much extra information and 
our models will try to fit over those extra bits and there will be problem of 
overfitting. It's better to choose only those columns that provide value and 
information that is important to predict arrival delays.
"""

print("\n=== Filtered Dataset with Selected Columns - First 5 rows ===")
print(df_filtered.head())

columns = df_filtered.columns
print('\nTotal columns - ' + str(len(columns)))

# ### Handling columns with value in 'hhmm' format

# Columns that are in the "hhmm" format
print("\n=== Time Columns (hhmm format) ===")
print(df_filtered[['CRSDepTime','DepTime','CRSArrTime','WheelsOff', 'WheelsOn']])

"""
We will be focusing on transforming time-related (hhmm) columns.

1. The code iterates over each column name in time_columns.

2. For each column, it divides the column value by 100 to extract the hour portion. 
   Similarly we calculate the remainder of the column value by dividing by 100 
   (% 100) to extract minutes.

3. We apply trigonometric transformations to convert the time into a cyclical 
   format using sine and cosine functions.

4. After the transformations, the original time column and the intermediate hour 
   and minute columns are dropped from the DataFrame, as they are no longer needed 
   in their original forms.

5. Finally, the code prints the data types of the updated DataFrame columns, 
   likely to verify the successful creation of the new columns and the removal 
   of the old ones.
"""

time_columns = ['CRSDepTime','DepTime','CRSArrTime','WheelsOff', 'WheelsOn']

for column in time_columns:
    # Extract hour and minute
    df_filtered[column + '_hour'] = df_filtered[column] // 100
    df_filtered[column + '_minute'] = df_filtered[column] % 100

    # Apply trigonometric transformations
    df_filtered[column + '_sin'] = np.sin(2 * np.pi * (df_filtered[column + '_hour'] * 60 + df_filtered[column + '_minute']) / (24 * 60))
    df_filtered[column + '_cos'] = np.cos(2 * np.pi * (df_filtered[column + '_hour'] * 60 + df_filtered[column + '_minute']) / (24 * 60))

    # Drop the original columns
    df_filtered = df_filtered.drop(columns=[column, column + '_hour', column + '_minute'])

# Print data types of the updated DataFrame
print("\n=== Data Types After Time Transformation ===")
print(df_filtered.dtypes)

# Check for duplicates and drop them

# Checking for duplicates
duplicates_count = df_filtered.duplicated().sum()
print(f"\n=== Duplicates Found: {duplicates_count} ===")

# Drop duplicates
df_filtered = df_filtered.drop_duplicates()
print(f"=== Duplicates Removed ===")

# ### Handling missing values in the dataset

# Showing missing values
missing_values = df_filtered.isnull().sum()
print("\n=== Missing Values ===")
print(missing_values)

# Count of unique values in target
print("\n=== Target Variable (ArrDel15) Distribution ===")
print(df_filtered['ArrDel15'].value_counts())

# ### Pipeline

"""
1. The provided code sets up a data preprocessing pipeline using make_pipeline 
   and ColumnTransformer.
   
2. It identifies numerical and categorical columns, removes the target columns, 
   and creates separate pipelines for numerical and categorical features.
   
3. The num_pipeline imputes missing values and standardizes numerical features, 
   while the cat_pipeline imputes missing values and one-hot encodes categorical 
   features.
   
4. The preprocessor combines both pipelines. This preprocessing ensures consistent 
   feature scaling and handling of missing data before training machine learning models.
"""

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define numerical and categorical columns
num_cols = df_filtered.select_dtypes(include='number').columns.to_list()
cat_cols = df_filtered.select_dtypes(exclude='number').columns.to_list()

# Remove the target column from numerical columns
num_cols.remove('DepDel15')
num_cols.remove('ArrDel15')

# Create pipelines for numerical and categorical columns
num_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())

# Use ColumnTransformer to set the estimators and transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ],
    remainder='passthrough'
)

print("\n=== Preprocessor Pipeline ===")
print(preprocessor)

"""
Numerical Pipeline:
1. SimpleImputer: Missing values are filled with the mean value of the feature.
2. StandardScaler: Numerical features are scaled to have zero mean and unit variance.

Categorical Pipeline:
1. SimpleImputer: Missing values are filled with the most frequent value (mode) 
   of the feature.
2. OneHotEncoder: Categorical features are transformed into binary numbers 
   (for example: 001,110).
"""

# Apply the preprocessing pipeline on the dataset
df_prepared = preprocessor.fit_transform(df_filtered)

# Scikit-learn strips the column headers in most cases, so just add them back on afterward.
feature_names = preprocessor.get_feature_names_out()

df_prepared = pd.DataFrame(data=df_prepared, columns=feature_names)

print("\n=== Prepared Dataset - First 5 rows ===")
print(df_prepared.head())

# Verifying missing values
missing_values_count = df_prepared.isnull().sum()
print("\n=== Missing Values Count for Each Column (After Preprocessing) ===")
print(missing_values_count)

"""
We drop the missing values from the 'remainder__ArrDel15' and 'remainder__DepDel15' 
columns. We cannot fill them up with most frequent item because 'remainder__ArrDel15' 
is target column and for 'remainder__DepDel15' there is a chance that it might 
create a biased model.
"""

df_prepared = df_prepared.dropna(subset=['remainder__ArrDel15', 'remainder__DepDel15'])

# Verifying missing values
missing_values_count = df_prepared.isnull().sum()
print("\n=== Missing Values Count After Dropping NaN (Target Columns) ===")
print(missing_values_count)

# Data types of the columns
data_types = df_prepared.dtypes
print("\n=== Data Types of Columns ===")
print(data_types)

"""
Notes:

- Columns prefixed with num__ are numerical features.
- Columns prefixed with cat__ are one-hot encoded categorical features.
- The column 'remainder__ArrDel15', 'remainder__DepDel15' are the binary features 
  for ArrDel15 and DepDel15 respectively.

The columns representing binary features (ArrDel15 and DepDel15) should ideally 
be of integer type (0 or 1) instead of float. We convert these columns to integer 
type as they are supposed to represent binary values.
"""

# Converting target column to Int
# Convert binary columns to integer type
df_prepared['remainder__ArrDel15'] = df_prepared['remainder__ArrDel15'].astype(int)
df_prepared['remainder__DepDel15'] = df_prepared['remainder__DepDel15'].astype(int)

print(f"\n=== Prepared Dataset Shape: {df_prepared.shape} ===")

# ==============================================================================
# 5. Selecting the models for training
# ==============================================================================

print("\n" + "="*80)
print("MODEL TRAINING")
print("="*80)

# ### Splitting the dataset for training and testing

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define X (features) and y (target)
X = df_prepared.drop(['remainder__ArrDel15'],axis=1)
y = df_prepared['remainder__ArrDel15']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n=== Train/Test Split ===")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ### Random Forest Classifier

print("\n--- Training Random Forest Classifier ---")
# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)

# ### Logistic Regression

print("\n--- Training Logistic Regression ---")
# Train Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_accuracy)

# ### kNN Classifier

print("\n--- Training kNN Classifier ---")
from sklearn.neighbors import KNeighborsClassifier

# Ensure X_test is a numpy array and is C-contiguous
X_test_array = np.asarray(X_test, order='C')

knn_model = KNeighborsClassifier(n_neighbors=3) 
knn_model.fit(X_train, y_train)

# Attempt prediction again with the modified X_test
knn_pred = knn_model.predict(X_test_array)
knn_accuracy = accuracy_score(y_test, knn_pred)

print("kNN Accuracy:", knn_accuracy)

# ==============================================================================
# Evaluating the models
# ==============================================================================

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# ### Confusion Matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix

# Define function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    # Save figure
    figure_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {figure_path}")
    plt.close()

# Plot confusion matrix for kNN model
plot_confusion_matrix(y_test, knn_pred, 
                     title='Confusion Matrix - kNN Model',
                     filename='confusion_matrix_knn.png')

# Plot confusion matrix for Logistic Regression model
plot_confusion_matrix(y_test, lr_pred, 
                     title='Confusion Matrix - Logistic Regression Model',
                     filename='confusion_matrix_logistic_regression.png')

# Plot confusion matrix for Random Forest model
plot_confusion_matrix(y_test, rf_pred, 
                     title='Confusion Matrix - Random Forest Model',
                     filename='confusion_matrix_random_forest.png')

# ### Classification Report

# Classification report for kNN model
print("\n=== Classification Report - kNN Model ===")
print(classification_report(y_test, knn_pred))

# Classification report for Logistic Regression model
print("\n=== Classification Report - Logistic Regression Model ===")
print(classification_report(y_test, lr_pred))   

# Classification report for Random Forest model
print("\n=== Classification Report - Random Forest Model ===")
print(classification_report(y_test, rf_pred))

# ### Plotting bar chart for performance Metrics for Each Model

# Data
data = {
    'Model': ['kNN', 'Logistic Regression', 'Random Forest'],
    'Precision': [0.95, 0.96, 0.97],
    'Recall': [0.83, 0.82, 0.92],
    'F1-score': [0.81, 0.83, 0.91],
    'Accuracy': [0.93, 0.93, 0.96]
}

df_plot = pd.DataFrame(data)

# Melt the DataFrame
df_melted = pd.melt(df_plot, id_vars='Model', var_name='Metric', value_name='Score')

# Plot
plt.figure(figsize=(8, 6))
sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric')
plt.title('Performance Metrics for Each Model')
plt.ylabel('Score')
plt.xlabel('Model')
plt.legend(title='Metric')
plt.tight_layout()
# Save figure
figure_path = os.path.join(OUTPUT_DIR, 'performance_metrics_comparison.png')
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"Figure saved: {figure_path}")
plt.close()

# ### Plotting ROC curve

from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, rf_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.tight_layout()
# Save figure
figure_path = os.path.join(OUTPUT_DIR, 'roc_curve.png')
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"Figure saved: {figure_path}")
plt.close()

# ### Plotting precision-recall curve

from sklearn.metrics import precision_recall_curve

# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, rf_pred)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.tight_layout()
# Save figure
figure_path = os.path.join(OUTPUT_DIR, 'precision_recall_curve.png')
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"Figure saved: {figure_path}")
plt.close()

"""
Model Performance Summary:

- High Precision (0.97 for class 0 and 0.92 for class 1): This means the model 
  rarely predicts a positive case that's actually negative. There are very few 
  false positives.
  
- High Recall (0.98 for class 0 and 0.89 for class 1): This means the model 
  identifies most of the actual positive cases. There are few false negatives.
  
- High Accuracy (0.96): This indicates the model performs well on both positive 
  and negative classifications.
  
- But there might be a chance that model might be overfitting. We need to check 
  this and tune it accordingly.
"""

# ==============================================================================
# 6. Fine-Tuning Model
# ==============================================================================

print("\n" + "="*80)
print("MODEL FINE-TUNING")
print("="*80)

# ### RandomizedSearchCV

"""
Performing RandomizedSearchCV tuning on the Random Forest model using the 
specified parameter grid, and then evaluate the best model found by the search 
on the test data.
"""

from sklearn.model_selection import RandomizedSearchCV

# Define a smaller parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}

# Create a Random Forest classifier
rf_model = RandomForestClassifier()

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid, n_iter=10,
                                   scoring='accuracy', cv=5, verbose=2, random_state=42, n_jobs=-1)

print("\n--- Starting RandomizedSearchCV ---")
print("This may take several minutes...")

# Fit the RandomizedSearchCV object to the full training data
random_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = random_search.best_params_
best_model = random_search.best_estimator_

print("\n=== Best Parameters ===")
print(best_params)

# Evaluate the best model on the test data
best_pred = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, best_pred)
print("\n=== Best Random Forest Accuracy ===")
print(f"Accuracy: {best_accuracy}")

# ==============================================================================
# Final Summary
# ==============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nAll visualizations have been saved to:", os.path.abspath(OUTPUT_DIR))
print("\nGenerated figures:")
print("1. top10_busiest_airports.png - Top 10 busiest airports visualization")
print("2. average_delays_by_hour.png - Average flight delays by time of day")
print("3. correlation_matrix.png - Correlation matrix of numerical features")
print("4. confusion_matrix_knn.png - Confusion matrix for kNN model")
print("5. confusion_matrix_logistic_regression.png - Confusion matrix for Logistic Regression")
print("6. confusion_matrix_random_forest.png - Confusion matrix for Random Forest")
print("7. performance_metrics_comparison.png - Performance metrics comparison")
print("8. roc_curve.png - ROC curve for Random Forest model")
print("9. precision_recall_curve.png - Precision-Recall curve")
print("\nModel Training Complete!")
print("="*80)

