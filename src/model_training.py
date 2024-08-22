import pandas as pd
import xgboost as xgb
import joblib

def train_model(X_train, y_train):
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Set XGBoost parameters
    params = {
        'max_depth': 6,
        'eta': 0.3,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'num_class': 1
    }

    # Train XGBoost model
    num_round = 100
    model = xgb.train(params, dtrain, num_round)
    
    return model

if __name__ == "__main__":
    X_train = pd.read_csv("data/X_train_processed.csv")
    y_train = pd.read_csv("data/y_train.csv")

    # Ensure y_train is numeric
    y_train = y_train['Label'].astype(int)

    model = train_model(X_train, y_train)
    joblib.dump(model, "models/fraud_detection_model.pkl")
    print("Model training completed.")