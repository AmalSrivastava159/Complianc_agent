import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 1️⃣ Load your transaction dataset from the local path
try:
    df = pd.read_excel("user_monthly_transactions_with_consent.xlsx")
    print("✅ Successfully loaded user_monthly_transactions_with_consent.xlsx")
except FileNotFoundError:
    print("❌ Error: 'user_monthly_transactions_with_consent.xlsx' not found.")
    print("Please make sure the Excel file is in the same directory as this script.")
    exit()

# 2️⃣ Define high-risk countries for AML check
high_risk_countries = ["Pakistan", "Afghanistan", "Syria", "North Korea"]

# 3️⃣ Simulate receiver country (since your dataset has only cities, we mock it for testing)
# In production, map city to country using a dictionary or actual data.
np.random.seed(42)
df["receiver_country"] = np.random.choice(
    ["India", "Pakistan", "Syria", "Afghanistan"],
    size=len(df),
    p=[0.85, 0.05, 0.05, 0.05]
)

# 4️⃣ Feature: Is transaction from/to a high-risk country
df["is_high_risk_country"] = df["receiver_country"].isin(high_risk_countries).astype(int)

# 5️⃣ Feature: Is cross-city transaction
df["is_cross_city"] = (df["sender_location"] != df["receiver_location"]).astype(int)

# 6️⃣ Feature: Is transaction > 3x user's weekly expense
# Approximate weekly expense as 10% of user's mean transaction value
user_weekly_avg = df.groupby("sender_account")["amount"].transform("mean") * 0.1
df["is_large_transaction"] = (df["amount"] > 3 * user_weekly_avg).astype(int)

# 7️⃣ Target: is_fraud
df["is_fraud"] = 0

# If 'Without Consent' -> mark as fraud
df.loc[df["consent_status"] == "Without Consent", "is_fraud"] = 1

# If 'With Consent' -> check fraud conditions
consent_mask = df["consent_status"] == "With Consent"
df.loc[
    consent_mask & (
        (df["is_high_risk_country"] == 1) |
        (df["is_cross_city"] == 1) |
        (df["is_large_transaction"] == 1)
    ),
    "is_fraud"
] = 1

# 8️⃣ Model Training
features = ["is_high_risk_country", "is_cross_city", "is_large_transaction", "amount"]
X = df[features]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# 9️⃣ Evaluation
y_pred = model.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 1️⃣0️⃣ Save the trained model for later use
joblib.dump(model, "aml_fraud_detection_rf_model.pkl")
print("\n✅ Model training complete and saved as 'aml_fraud_detection_rf_model.pkl'")

# Optional: View sample predictions
# sample_preds = X_test.copy()
# sample_preds["predicted_is_fraud"] = y_pred
# sample_preds["actual_is_fraud"] = y_test.values
# print("\n📋 Sample Predictions:")
# print(sample_preds.head())
