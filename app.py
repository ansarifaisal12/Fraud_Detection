import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Load the model and preprocessor
model = joblib.load("models/fraud_detection_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

st.title('Fraud Detection Application')

st.write("""
This app predicts the probability of a transaction being fraudulent.
Please input the transaction details below.
""")

# Create input fields for user
transaction_amount = st.number_input('Transaction Amount', min_value=0.0, format='%f')
transaction_duration = st.number_input('Transaction Duration (seconds)', min_value=0.0, format='%f')
user_age = st.number_input('User Age', min_value=0, max_value=120)
merchant_category = st.selectbox('Merchant Category', preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[0])
location = st.selectbox('Location', preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[1])
time_of_day = st.selectbox('Time of Day', preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[2])
user_gender = st.selectbox('User Gender', preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[3])

if st.button('Predict'):
    # Create a dictionary with user inputs
    input_data = {
        'TransactionAmount': [transaction_amount],
        'TransactionDuration': [transaction_duration],
        'UserAge': [user_age],
        'MerchantCategory': [merchant_category],
        'Location': [location],
        'TimeOfDay': [time_of_day],
        'UserGender': [user_gender]
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_data)

    # Preprocess the input data
    processed_data = preprocessor.transform(input_df)

    # Convert to DataFrame
    feature_names = (
        ['TransactionAmount', 'TransactionDuration', 'UserAge'] +
        [f'MerchantCategory_{cat}' for cat in preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[0][1:]] +
        [f'Location_{cat}' for cat in preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[1][1:]] +
        [f'TimeOfDay_{cat}' for cat in preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[2][1:]] +
        [f'UserGender_{cat}' for cat in preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[3][1:]]
    )
    processed_df = pd.DataFrame(processed_data, columns=feature_names)

    # Create DMatrix for XGBoost
    dmatrix = xgb.DMatrix(processed_df)
    
    # Make prediction
    prediction_proba = model.predict(dmatrix)[0]
    prediction = 'Fraudulent' if prediction_proba > 0.5 else 'Legitimate'
    
    st.write(f'Prediction: {prediction}')
    st.write(f'Probability of fraud: {prediction_proba:.2%}')

    # Visualize the prediction
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.bar(['Legitimate', 'Fraudulent'], [1-prediction_proba, prediction_proba])
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')
    st.pyplot(fig)

st.write("""
Note: This is a demonstration model and should not be used for actual fraud detection without further validation and testing.
""")