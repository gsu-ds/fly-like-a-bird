# Model Selection, Training, and Evaluation for Flight Status Prediction

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import joblib
import wandb

wandb.init(project="Data Mining - Senior Project")


OUTPUT_DIR = '/workspaces/fly-like-a-bird/flight_status_prediction/reports/figures/late_runs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv('/workspaces/fly-like-a-bird/flight_status_prediction/data/processed/cleaned_data.csv')  

# Define X (features) and y (target)
X = df.drop(['remainder__ArrDel15'],axis=1)
y = df['remainder__ArrDel15']

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


# Ensure X_test is a numpy array and is C-contiguous
X_test_array = np.asarray(X_test, order='C')

knn_model = KNeighborsClassifier(n_neighbors=3) 
knn_model.fit(X_train, y_train)

# Attempt prediction again with the modified X_test
knn_pred = knn_model.predict(X_test_array)
knn_accuracy = accuracy_score(y_test, knn_pred)

print("kNN Accuracy:", knn_accuracy)


# Random Forest Metrics
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

# Logistic Regression Metrics
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

# kNN Metrics
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)

# Log metrics to Wandb
wandb.log({
    "rf_initial_accuracy": rf_accuracy,
    "lr_initial_accuracy": lr_accuracy,
    "knn_initial_accuracy": knn_accuracy
})


# Evaluation of Models

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# ### Confusion Matrix


# Define function to plot confusion matrix (updated to use wandb)
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # 🌟 WANDB LOGGING: Log the figure before saving/closing
    wandb.log({f"Confusion Matrix - {title}": wandb.Image(plt)}) 

    # Save figure (keeping this for local file output, but not necessary for W&B)
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
    'Precision': [knn_precision, lr_precision, rf_precision],
    'Recall': [knn_recall, lr_recall, rf_recall],
    'F1-score': [knn_f1, lr_f1, rf_f1],
    'Accuracy': [knn_accuracy, lr_accuracy, rf_accuracy]
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
wandb.log({"Performance Metrics Comparison": wandb.Image(plt)})

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
wandb.log({"ROC Curve": wandb.Image(plt)})

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
wandb.log({"Precision-Recall Curve": wandb.Image(plt)})


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
wandb.log({
    "best_rf_params": best_params,
    "best_rf_accuracy": best_accuracy, 
})

# Saving model as an artifact in Wandb

model_output_path = "best_rf_model.pkl"
joblib.dump(best_model, model_output_path)

model_artifact = wandb.Artifact(
    name="best_random_forest_model", 
    type="model",
    description="The fine-tuned Random Forest model."
)
model_artifact.add_file(model_output_path)
wandb.log_artifact(model_artifact)

# Wandb closeout
wandb.finish()


# Summary of all generated visualizations

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

