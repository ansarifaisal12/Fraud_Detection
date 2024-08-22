import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Convert 'Label' to numeric: 'Fraudulent' -> 1, 'Legitimate' -> 0
    df['Label'] = (df['Label'] == 'Fraudulent').astype(int)
    return df

def split_data(df, test_size=0.2, random_state=42):
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data("data/transaction_fraud_data.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Save split data
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    
    print("Data ingestion completed.")