import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

print("Loading dataset...")
# 1. Load dataset spreadsheet
df = pd.read_csv('transaction_dataset.csv')

# Clean up column names: strips trailing spaces and standardizes capitalization
df.columns = df.columns.str.strip()

# Find the exact column names regardless of capitalization
cols_lower = {col.lower(): col for col in df.columns}

# Handle 'Total Ether Sent' variations safely
sent_col = cols_lower.get('total ether sent') or cols_lower.get('total ether sent ') or 'Total Ether sent'
recv_col = cols_lower.get('total ether received') or cols_lower.get('total ether received ') or 'Total Ether received'

# 2. Map and select key behavioral features directly from dataframe columns
df = df.rename(columns={
    'Unique Received From Addresses': 'Unique_Received_From_Addresses',
    'Unique Sent To Addresses': 'Unique_Sent_To_Addresses',
    sent_col: 'Total_Ether_Sent',
    recv_col: 'Total_Ether_Received'
})

# Standardize spaces to underscores across all columns
df.columns = [c.replace(' ', '_') for c in df.columns]

features = [
    'Avg_min_between_sent_tnx',
    'Avg_min_between_received_tnx',
    'Time_Diff_between_first_and_last_(Mins)',
    'Sent_tnx',
    'Received_Tnx',
    'Unique_Received_From_Addresses',
    'Unique_Sent_To_Addresses',
    'Total_Ether_Sent',
    'Total_Ether_Received'
]

# Separate features (X) and answers/fraud flags (y)
X = df[features].fillna(0)
y = df['FLAG']  # 1 = Fraud, 0 = Non-Fraud

# 3. Split data into Training set (80%) and Test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Random Forest AI model...")
# 4. Train Model using Scikit-Learn Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred = model.predict(X_test)
print("\n--- MODEL ACCURACY ---")
print("F1 Score:", f1_score(y_test, y_pred))

# 6. Save model to disk
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(features, 'model_features.pkl')
print("\nSuccess! AI model saved as 'fraud_model.pkl' inside your folder!")