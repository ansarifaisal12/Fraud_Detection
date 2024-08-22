import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

def evaluate_model(model, X_test, y_test):
    # Create DMatrix for XGBoost
    dtest = xgb.DMatrix(X_test)
    
    # Make predictions
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    model = joblib.load("models/fraud_detection_model.pkl")
    X_test = pd.read_csv("data/X_test_processed.csv")
    y_test = pd.read_csv("data/y_test.csv")

    # Ensure y_test is numeric
    y_test = y_test['Label'].astype(int)

    evaluate_model(model, X_test, y_test)
    print("Model evaluation completed.")