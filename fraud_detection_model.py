import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE

from sqlalchemy import create_engine


# Load the CSV
csv_path = "C:/Users/hp/Desktop/projects/Fraud Detection/creditcard.csv"
df = pd.read_csv(csv_path)

# Connect to PostgreSQL
engine = create_engine("postgresql://postgres:123456@localhost:5432/fraud_detection")

# Write to PostgreSQL
from sqlalchemy.exc import SQLAlchemyError

try:
    with engine.begin() as connection:
        df.to_sql("transactions", connection, index=False, if_exists="replace")
    print(f"✅ Successfully imported {len(df)} rows.")
except SQLAlchemyError as e:
    print("❌ Import failed:", e)


if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

df = pd.read_sql("SELECT * FROM transactions", con=engine)


# Features and target
features = [col for col in df.columns if col not in ['Class']]
target = 'Class'

X = df[features]
y = df[target]

# Check class balance
print("\nClass Distribution:")
print(y.value_counts(normalize=True))

# Preprocessing: scale 'Amount' and 'Time', leave PCA features as is
scaler = StandardScaler()
X = X.copy()  # Ensure you're working with a copy of the DataFrame
X.loc[:, ['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_res, y_train_res)

# Predict
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

# Feature importances
feature_names = X.columns
importances = clf.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()

# Combine predictions with test data
results_df = X_test.copy()
results_df['actual'] = y_test.values
results_df['predicted'] = y_pred
results_df['fraud_probability'] = y_proba

# Export to CSV
results_df.to_csv("fraud_predictions.csv", index=False)
print("✅ Exported predictions to fraud_predictions.csv")

