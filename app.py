import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Loading the trained model
model = tf.keras.models.load_model('model.h5')

# Load the Encoder and Scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)








# streamlit app
st.title('Customer Churn Prediction')


# ---- User Input ---- #

# Geography (OneHotEncoder)
geography = st.selectbox(
    'Geography',
    list(onehot_encoder_geo.categories_[0])
)

# Gender (LabelEncoder) - cleaner UI + safe encoding
gender = st.selectbox(
    'Gender',
    ['Male', 'Female']  
)
gender_encoded = label_encoder_gender.transform([gender])[0]

# Numeric Inputs
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

estimated_salary = st.number_input(
    'Estimated Salary',
    min_value=0.0,
    step=1000.0,
    format="%.2f"
)

tenure = st.slider('Tenure (Years)', 0, 10, 3)

num_of_products = st.slider('Number of Products', 1, 4, 1)

# Binary Inputs (Better UX with Yes/No)
has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
has_cr_card = 1 if has_cr_card == 'Yes' else 0

is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])
is_active_member = 1 if is_active_member == 'Yes' else 0


input_data = pd.DataFrame({
    "CreditScore" : [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns= onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop= True), geo_encoded_df], axis = 1)

input_data_scaled = scaler.transform(input_data)

# Prediction Churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write('probability: ', prediction_proba)

if prediction_proba > 0.5:
    st.write('The Customer is likely to churn')
else:
    st.write("The Customer is not likely to churn")