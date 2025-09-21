import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os

print("🚀 Starting model training and data preprocessing...")

# Example dataset path
filename = "Emp_data((3).csv"

if not os.path.exists(filename):
    print(f"❌ Error: The file '{filename}' was not found.")
    exit()

# Load dataset
emp_data = pd.read_csv(filename)

# Encode categorical columns
le_dropout = LabelEncoder()
emp_data['dropout'] = le_dropout.fit_transform(emp_data['dropout'])

if 'backlog' in emp_data.columns and emp_data['backlog'].dtype == 'object':
    emp_data['backlog'] = LabelEncoder().fit_transform(emp_data['backlog'])

# Features & target
X = emp_data.drop(['dropout', 'st_id', 'name'], axis=1, errors='ignore')
y = emp_data['dropout'].astype(int)

# Handle missing values
X = X.fillna(X.mean())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, '../model/feature_names.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Balance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)

# Evaluate
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n📊 Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Save artifacts
joblib.dump(rf_model, '../model/rf_model.pkl')
joblib.dump(scaler, '../model/scaler.pkl')
joblib.dump(le_dropout, '../model/le_dropout.pkl')

print("\n✅ Model training complete. Files saved in /model")
