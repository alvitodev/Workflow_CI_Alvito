import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

print("Memulai proses training dari dalam MLflow Project...")

# 1. Memuat Data dari folder telco_preprocessing
train_path = os.path.join("telco_preprocessing", "train.csv")
test_path = os.path.join("telco_preprocessing", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']
X_test = test_df.drop('Churn', axis=1)
y_test = test_df['Churn']

# 2. Mengaktifkan Autolog agar model dan signature otomatis tersimpan
mlflow.autolog()

# 3. Setup dan Training Model
with mlflow.start_run() as run:
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Training Selesai! Akurasi Model: {acc:.4f}")