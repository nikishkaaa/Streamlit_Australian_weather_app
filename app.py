import numpy as np
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import time

st.set_page_config(layout="wide")

st.title("Weather Forecasting Model: Will it Rain Tomorrow?")
st.markdown(" #### This app predicts whether it will rain tomorrow based on today's data.")
image = Image.open('images/rain.jpeg')
desired_width = 500  
st.image(image, width=desired_width)

st.divider()
st.markdown(" ### Enter today's weather conditions:")


# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('model/aussie_rain_pipeline.joblib')

model = load_model()

# Load the dataset
@st.cache_resource
def load_data():
    return pd.read_csv('data/weatherAUS.csv')

data = load_data()

# Calculate min and max values from your dataset
min_temp_min = data['MinTemp'].min()
min_temp_max = data['MinTemp'].max()

max_temp_min = data['MaxTemp'].min()
max_temp_max = data['MaxTemp'].max()

rainfall_min = data['Rainfall'].min()
rainfall_max = data['Rainfall'].max()

evaporation_min = data['Evaporation'].min()
evaporation_max = data['Evaporation'].max()

sunshine_min = data['Sunshine'].min()
sunshine_max = data['Sunshine'].max()

cloud_3pm_min = data['Cloud3pm'].min()
cloud_3pm_max = data['Cloud3pm'].max()

temp_9am_min = data['Temp9am'].min()
temp_9am_max = data['Temp9am'].max()

temp_3pm_min = data['Temp3pm'].min()
temp_3pm_max = data['Temp3pm'].max()

def predict_weather(input_data):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1][0]
    return prediction[0], probability

# Create columns to organize input fields
col1, col2, col3 = st.columns(3)

with col1:
    min_temp = st.number_input('Min Temperature (°C)', value=min_temp_min, min_value=min_temp_min, max_value=min_temp_max, step=0.1)
    max_temp = st.number_input('Max Temperature (°C)', value=max_temp_min, min_value=max_temp_min, max_value=max_temp_max, step=0.1)
    rainfall = st.number_input('Rainfall (mm)', value=rainfall_min, min_value=rainfall_min, max_value=rainfall_max, step=0.1)

with col2:
    evaporation = st.number_input('Evaporation (mm)', value=evaporation_min, min_value=evaporation_min, max_value=evaporation_max, step=0.1)
    sunshine = st.number_input('Sunshine (hrs)', value=sunshine_min, min_value=sunshine_min, max_value=sunshine_max, step=0.1)
    cloud_3pm = st.number_input('Cloudiness at 3pm (oktas)', value=cloud_3pm_min, min_value=cloud_3pm_min,  max_value=cloud_3pm_max, step=1.0)

with col3:
    temp_9am = st.number_input('Temperature at 9am (°C)', value=temp_9am_min, min_value=temp_9am_min, max_value=temp_9am_max, step=0.1)
    temp_3pm = st.number_input('Temperature at 3pm (°C)', value=temp_3pm_min, min_value=temp_3pm_min, max_value=temp_3pm_max, step=0.1)
    rain_today = st.selectbox('Was it raining today?', ['Yes', 'No'])

# Add 'RainToday' to the input data
data = pd.DataFrame({
    'MinTemp': [min_temp],
    'MaxTemp': [max_temp],
    'Rainfall': [rainfall],
    'Evaporation': [evaporation],
    'Sunshine': [sunshine],
    'Cloud3pm': [cloud_3pm],
    'Temp9am': [temp_9am],
    'Temp3pm': [temp_3pm],
    'RainToday': [rain_today]  # Pass 'Yes' or 'No' directly
})

st.markdown(
    """
     <style>
        .stButton > button {
            background-color: #007BFF; 
            color: white;                 
            width: 100%;                 
            height: 50px;                 
            font-size: 16px;             
        }
        .button-container {
            display: flex;
            justify-content: center;     
            margin: 20px 0;              
        }
        .prediction-result, .probability-result {
            text-align: center;          
            font-size: 24px;             
            margin: 20px 0;          
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Button to predict with progress bar
if st.button("Predict Rain"):
    with st.spinner('Predicting...'):
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        prediction, probability = model.predict(data), model.predict_proba(data)[:, 1][0]

    # Displaying the result with improved formatting and centered text
    st.markdown(f"<div class='prediction-result'>Will it rain tomorrow? {'Yes' if prediction == 'Yes' else 'No'}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='probability-result'>Prediction Probability: {probability:.2f}</div>", unsafe_allow_html=True)
