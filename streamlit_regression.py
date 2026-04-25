import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load model
model = tf.keras.models.load_model('regression_model.h5')

# Load encoders & scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)


# ---------------- STREAMLIT UI ---------------- #

st.title('Estimated Salary Prediction')

# ---- User Input ---- #

geography = st.selectbox(
    'Geography',
    list(onehot_encoder_geo.categories_[0])
)

gender = st.selectbox('Gender', ['Male', 'Female'])
gender_encoded = label_encoder_gender.transform([gender])[0]

age = st.slider('Age', 18, 92, 30)

balance = st.number_input(
    'Balance',
    min_value=0.0,
    step=100.0,
    format="%.2f"
)

credit_score = st.number_input(
    'Credit Score',
    min_value=300,
    max_value=900,
    value=650
)

tenure = st.slider('Tenure (Years)', 0, 10, 3)

num_of_products = st.slider('Number of Products', 1, 4, 1)

has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
has_cr_card = 1 if has_cr_card == 'Yes' else 0

is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])
is_active_member = 1 if is_active_member == 'Yes' else 0

exited = st.selectbox("Exited", [0, 1])


# ---------------- DATA PREPARATION ---------------- #

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "Exited": [exited]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_columns = onehot_encoder_geo.get_feature_names_out(['Geography'])

geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_columns)

# Combine all features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Match training columns
input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scale input
input_data_scaled = scaler.transform(input_data)


# ---------------- PREDICTION ---------------- #

prediction = model.predict(input_data_scaled)
predicted_salary = float(prediction[0][0])

st.subheader(f'Estimated Salary: {predicted_salary:,.2f}')