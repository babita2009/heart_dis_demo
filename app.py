import streamlit as st
import pandas as pd 
import joblib
import numpy as np
heart_model = joblib.load('heart_disease_model.pkl')
heart_scaler = joblib.load('scaler.pkl')
st.title('HEART DISEASE PREDICTION MODEL')
# st.image('https://www.heart.org/-/media/Images/Health-Topics/Answers-by-Heart-FAQs/What-is-heart-disease-answers-by-heart-faqs.jpg', caption='Heart Disease', use_column_width=True)

st.image('https://www.heart.org/-/media/Images/Health-Topics/Answers-by-Heart-FAQs/What-is-heart-disease-answers-by-heart-faqs.jpg', caption="human heart anatomy", width=250, use_container_width=False)     
width=250
use_container_width=False
age=st.slider('Age', 1, 120, 60)
anaemia = st.selectbox('Anaemia (1 = yes, 0 = no)', [0, 1])
creatinint= st.number_input('Creatinine Phosphokinase', 0, 8000)
diabetes = st.radio('Diabetes', (0, 1),format_func=lambda x: 'Yes' if x == 1 else 'No')
ef = st.number_input('Ejection Fraction', 10, 100)
hbp = st.radio('High Blood Pressure', (0, 1),format_func=lambda x: 'Yes' if x == 1 else 'No')
platelets = st.number_input('Platelets', 0.0, 1000000.0)
serum_creatinine = st.number_input('Serum Creatinine', 0.0, 10.0)
serum_sodium = st.number_input('Serum Sodium', 0, 200)
sex = st.selectbox('Sex (1 = male, 0 = female)', [0, 1])
smoking = st.radio('Smoking', [0, 1])
time = st.number_input('Time (in days)', 0, 300)
if st.button('heart failure'):
    heart_input = [[age, anaemia, creatinint, diabetes, ef, hbp, platelets,
                            serum_creatinine, serum_sodium, sex, smoking, time]]
    heart_input_scaled = heart_scaler.transform(heart_input)
    heart_prediction = heart_model.predict(heart_input_scaled)
    if heart_prediction[0]==1:
        st.error('The patient is likely to have a heart failure.')
    else:
        st.success('The patient is not likely to have a heart failure.')