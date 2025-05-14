
# Scaler is exported as scaler2.pkl
# Model is exported as model2.pkl 
# Order of the X 'Age', 'Balance' 'IsActiveMember' 


import streamlit as st 
import joblib 
import numpy as np 

scaler = joblib.load("C:/Users/rajis/Downloads/scaler2.pkl")
model = joblib.load("C:/Users/rajis/Downloads/model2.pkl")

st.title('Churn Prediction For Banks')

st.divider() 

st.write('Please enter the values and hit the predict button for getting a prediction')

st.divider()

age =st.number_input('Enter age', min_value =10, max_value =100, value = 30 )

Blanace = st.number_input('Enter Blanace', min_value=0, max_value = 1500)

Status = st.selectbox('Select Status',['IsActive','IsNotActive'])

st.divider()

predictbutton = st.button ('predict')

if predictbutton:
    Active_selected = 1 if Status == 'IsActive' else 0

    x = [age, Blanace, Active_selected]

    x_np = np.array([x])  
    x_scaled = scaler.transform(x_np)

    proba = model.predict_proba(x_scaled)[0][1]  

    predicted = 'Yes' if proba > 0.3 else 'No'
    st.write(f'Prediction: {predicted} with probability {proba:.2f}')

else:
    st.write('Please enter the values and use the predict button')
