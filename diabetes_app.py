import streamlit as st
import numpy as np
import pickle

# Load the model

rfc_pickle = open('random_forest_classifier_model.pkl', 'rb')
model = pickle.load(rfc_pickle)

# Define a function to make predictions
def make_prediction(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]).astype('float64')
    prediction = model.predict(input_data)
    return prediction[0]


# Sidebar for user input
st.sidebar.header('User Input Parameters')

pregnancies = st.sidebar.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input('Glucose Level', min_value=0.0, max_value=200.0, value=120.0)
blood_pressure = st.sidebar.number_input('Blood Pressure', min_value=0.0, max_value=120.0, value=70.0)
skin_thickness = st.sidebar.number_input('Skin Thickness', min_value=0.0, max_value=100.0, value=20.0)
insulin = st.sidebar.number_input('Insulin Level', min_value=0.0, max_value=200.0, value=70.0)
bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=60.0, value=30.0)
diabetes_pedigree_function = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.0, value=0.5)
age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30)

if st.sidebar.button('Predict'):
    prediction = make_prediction(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    if prediction == 1:
        st.success('The model predicts: Yes, you have diabetes')
    else:
        st.success('The model predicts: No, you do not have diabetes')

# Streamlit app layout
st.title('Diabetes Prediction Model')
st.subheader('An App by Jonathan Ibifubara Pollyn')
st.write('This application is used to make predictions of diabetes based on known features. The purpose of this exercise is to demonstrate the power of Machine learning. The dataset contains a range of health-related attributes collected to aid the development of predictive models to identify any risk of diabetes. I will build the model and indicate each phase that was undergoing. Ultimately, I will write detailed documentation explaining everything that happened and my findings. The inspiration for this work is due to my passion for working in the healthcare field and my data science knowledge. Below are details of this data and columns, along with a link to the data for more information.')

st.write("""
## Data Description

The dataset contains the following features:

- **Pregnancies**: Number of times pregnant.
- **Glucose**: Plasma glucose concentration over 2 hours in an oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg / height in m^2).
- **DiabetesPedigreeFunction**: Diabetes pedigree function, a genetic score of diabetes.
- **Age**: Age in years.
- **Outcome**: Binary classification indicating the presence (1) or absence (0) of diabetes.

You can access the dataset [here](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes).
""")
