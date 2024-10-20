# Streamlit_Australian_weather_app
## Weather forecasting model with Streamlit

##

## 📊 Dataset:
The model is trained on the Australian weather dataset from Kaggle. It includes multiple weather features collected across Australia, such as temperature, rainfall, and humidity.
Link to the dataset (https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

## 📁 Project structure:
 - data/: Directory containing the dataset.
 - images/: Directory for storing images used in the application.
 - model/: Directory containing the trained ML model.
 - app.py: Main file of the Streamlit application.
 - weather.ipynb: Jupyter Notebook for training the Random Forest model.


## 📝 Overview:
The app allows users to input various weather conditions such as temperature, rainfall, sunshine, and cloudiness to predict whether it will rain tomorrow. The model outputs:
- Rain Prediction (Yes/No)
- Prediction Probability: A confidence score indicating the likelihood of rain.
You can try the app here: https://app-australian-weather.streamlit.app/

## ⚙️ Tools & Technologies:
 - Python: Core language used in the project.
 - Scikit-learn: Used for building and training the machine learning model.
 - Pandas: For data manipulation and preprocessing.
 - Joblib: For model saving and loading.
 - Streamlit: For building an interactive web interface.
 - requirements.txt: List of necessary Python packages.
