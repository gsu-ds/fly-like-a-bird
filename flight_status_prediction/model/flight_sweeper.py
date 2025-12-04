# sweep_tuning.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import wandb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve

# ----------------------------------------------------
# 0. CONFIGURATION AND INITIAL SETUP
# ----------------------------------------------------

# Define the Hyperparameter Search Space for Random Forest
# We restrict max_depth and increase min_samples_leaf to combat overfitting
sweep_config = {
    'method': 'random',  # Random Search is generally efficient
    'metric': {
        'name': 'cv_accuracy',  # We will log the Cross-Validation Accuracy
        'goal': 'maximize'
    },
    'parameters': {
        # 'n_estimators': {
        #     'values': [50, 100, 150, 200, 250, 300]
        # },
        # Using distributions for more continuous sampling is often better for 'random' method:
        'n_estimators': {
             'distribution': 'int_uniform',
             'min': 50,
             'max': 300
        },
        'max_depth': {
            'values': [5, 10, 15, 20, 25, None] # Testing constrained depths and full depth
        },
        'min_samples_split': {
            'values': [2, 5, 10, 20]
        },
        'min_samples_leaf': {
            'values': [1, 2, 5, 10]
        },
        'criterion': {
            'values': ['gini', 'entropy']
        },
        'bootstrap': {
            'values': [True, False]
        }
    }
}

OUTPUT_DIR = '/workspaces/fly-like-a-bird/flight_status_prediction/reports/figures/late_runs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# 1. DATA LOADING AND SPLITTING (Global)
# ----------------------------------------------------

print("Loading and preparing data...")
df = pd.read_csv('/workspaces/fly-like-a-bird/flight_status_prediction/data/processed/cleaned_data.csv')

X = df.drop(['remainder__ArrDel15'],axis=1)
y = df['remainder__ArrDel15']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ----------------------------------------------------
# 2. TRAINING FUNCTION FOR W&B AGENT
# ----------------------------------------------------

def train_sweep_run():
    # Start a new run (trial)
    wandb.init(project="Data Mining - Senior Project")
    
    # Access the hyperparameters for this specific run
    config = wandb.config
    
    print(f"\n--- Starting run with parameters: {config} ---")

    # Initialize the Random Forest model with the sweep parameters
    rf_model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        criterion=config.criterion,
        bootstrap=config.bootstrap,
        n_jobs=-1, # Use all processors for speed
        random_state=42
    )

    # Use 5-fold Cross-Validation on the training set to evaluate the model
    # This gives a robust estimate of performance without touching the test set
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    
    mean_cv_accuracy = np.mean(cv_scores)
    
    # Log the key metric back to the W&B dashboard
    wandb.log({
        "cv_accuracy": mean_cv_accuracy,
        "cv_accuracy_std": np.std(cv_scores),
    })

    print(f"Cross-Validation Accuracy: {mean_cv_accuracy:.4f}")

    # Optionally, we can fit the model once on the full training set 
    # to evaluate the generalization on the held-out test set
    if mean_cv_accuracy > 0.90: # Only fit/log the better models to save time
        rf_model.fit(X_train, y_train)
        test_pred = rf_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        wandb.log({
            "test_accuracy": test_accuracy
        })
        
        print(f"Test Set Accuracy: {test_accuracy:.4f}")
        
        # Log the full classification report as text
        report = classification_report(y_test, test_pred, output_dict=True)
        wandb.log({"classification_report": wandb.Table(dataframe=pd.DataFrame(report).transpose())})


# ----------------------------------------------------
# 3. INITIALIZE AND RUN THE SWEEP
# ----------------------------------------------------

if __name__ == "__main__":
    # Create the sweep on the W&B server
    sweep_id = wandb.sweep(sweep_config, project="Data Mining - Senior Project")
    
    print("\n" + "="*80)
    print(f"W&B Sweep Initialized with ID: {sweep_id}")
    print("Go to the W&B Dashboard URL shown above to monitor results.")
    print("="*80)

    # Set 'count' to a high number (e.g., 200, 500) to run overnight.
    # Change the count based on how many hours you want it to run.
    wandb.agent(sweep_id, function=train_sweep_run, count=100) 
    
    print("\nSweep Complete!")
    print("W&B will automatically log the best run found across all agents.")