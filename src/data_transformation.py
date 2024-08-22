import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(X_train, X_test):
    # Define numeric and categorical columns
    numeric_features = ['TransactionAmount', 'TransactionDuration', 'UserAge']
    categorical_features = ['MerchantCategory', 'Location', 'TimeOfDay', 'UserGender']

    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit on the combined data
    X_combined = pd.concat([X_train, X_test], axis=0)
    preprocessor.fit(X_combined)

    # Transform train and test data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    numeric_feature_names = numeric_features
    categorical_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
    feature_names = numeric_feature_names + categorical_feature_names

    # Convert to DataFrames
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

    return X_train_processed, X_test_processed, preprocessor

if __name__ == "__main__":
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")

    X_train_processed, X_test_processed, preprocessor = preprocess_data(X_train, X_test)
    
    # Save processed data and preprocessor
    X_train_processed.to_csv("data/X_train_processed.csv", index=False)
    X_test_processed.to_csv("data/X_test_processed.csv", index=False)
    pd.to_pickle(preprocessor, "models/preprocessor.pkl")
    
    print("Data transformation completed.")