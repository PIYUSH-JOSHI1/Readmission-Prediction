# import
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
#________________________________________________________________________


def predict_readmission(Gender, Admission_Type, Diagnosis, Num_Lab_Procedures,
       Num_Medications, Num_Outpatient_Visits, Num_Inpatient_Visits,
       Num_Emergency_Visits, Num_Diagnoses, A1C_Result):

    with open("Readmission_Model.pkl","rb") as m:
        model = pickle.load(m)
    
    data = np.array([[Gender, Admission_Type, Diagnosis, Num_Lab_Procedures,
       Num_Medications, Num_Outpatient_Visits, Num_Inpatient_Visits,
       Num_Emergency_Visits, Num_Diagnoses, A1C_Result]])
    prediction = model.predict(data)
    out = prediction[0]
    return out 


#_________________________________________________________________________

st.set_page_config(page_title= "Predicting Hospital Readmissions",
                   layout= "wide",
                   menu_items={'About': "### This page is created by Desilva!"})

st.markdown("<h1 style='text-align: center; color: #fa6607;'>Predicting Hospital Readmissions</h1>", unsafe_allow_html=True)
st.write("")

select = option_menu(None,["Home", "Readmission"], 
                    icons =["hospital-fill","ticket-detailed"], orientation="horizontal",
                    styles={"container": {"padding": "0!important", "background-color": "#fafafa"},
                            "icon": {"color": "#fdfcfb", "font-size": "20px"},
                            "nav-link": {"font-size": "20px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
                            "nav-link-selected": {"background-color": "#fa6607"}})

if select == "Home":
    st.title("Welcome to the Hospital Readmissions Prediction Project!")

    st.write('''
**Objective:**
The primary goal of this project is to develop a predictive model that accurately determines whether a patient will require readmission within 30 days after their initial discharge. By leveraging advanced machine learning techniques, this project aims to enhance patient care and optimize healthcare resources.

**Key Features:**
- **Predictive Modeling:** This machine learning model analyzes patient data to predict the likelihood of hospital readmission within 30 days. This predictive capability enables proactive intervention and personalized care planning for at-risk patients.

- **Data-Driven Insights:** By processing comprehensive healthcare records, this model extracts valuable insights and identifies key factors contributing to readmission risk. These insights empower healthcare providers to make informed decisions and implement targeted interventions.

**How It Works:**
1. **Data Input:** This model accepts input data comprising patient demographics, medical history, previous hospitalizations, diagnoses, and medications.
   
2. **Predictive Analysis:** Leveraging state-of-the-art machine learning algorithms, the model processes the input data to generate predictions on whether readmission is required or not.

3. **Actionable Recommendations:** Based on the model's predictions, healthcare providers can proactively engage with high-risk patients, implement preventive measures, and optimize post-discharge care plans to minimize readmission rates.

**Technological Stack:**
This predictive model is developed using Python, a versatile programming language for data science and machine learning. It leverages industry-standard libraries such as scikit-learn for modeling and pandas for data manipulation to ensure robust performance and reliability.

**Stay Informed:**
Follow along as this project explores the intricacies of hospital readmissions prediction and strives to make a positive impact on patient outcomes and healthcare delivery.

''')

elif select == "Readmission":

    st.write("")
    st.header("Fill all the details below to know the prediction")
    st.write("")

    col1,col2,col3 = st.columns([5,1,5])
    with col1:
        selected_gender = st.selectbox('Select a Gender:', ["Female", "Male", "Other"])
        if selected_gender == "Female":
            Gender = 0
        elif selected_gender == "Male":
            Gender = 1
        else:
            Gender = 2


        Selected_Admission_Type  = st.selectbox('Select a Admission Type:', ['Emergency','Urgent', 'Elective'])
        if Selected_Admission_Type  == "Emergency":
            Admission_Type = 1
        elif Selected_Admission_Type == "Urgent":
            Admission_Type = 2
        else:
            Admission_Type = 0

        Selected_Diagnosis  = st.selectbox('Select a Diagnosis:', ['Heart Disease', 'Diabetes', 'Injury', 'Infection'])
        if Selected_Admission_Type  == "Heart Disease":
            Diagnosis = 1
        elif Selected_Admission_Type == "Diabetes":
            Diagnosis = 0
        elif Selected_Admission_Type == "Injury":
            Diagnosis = 3
        else:
            Diagnosis = 2

        Num_Lab_Procedures  = st.selectbox('Select a Number of Lab Procedures:', range(1,100))

        Num_Medications  = st.selectbox('Select a Number of Medications:', range(1,36))
                         
    with col3:
        Num_Outpatient_Visits  = st.selectbox('Select a Number of Outpatient Visits:', range(0,5))

        Num_Inpatient_Visits  = st.selectbox('Select a Number of Inpatient Visits:', range(0,5))

        Num_Emergency_Visits  = st.selectbox('Select a Number of Emergency Visits:', range(0,5))

        Num_Diagnoses  = st.selectbox('Select a Number of Diagnoses:', range(1,10))

        A1C = st.selectbox('Select a Number of A1C Result:', ['Normal','Abnormal'])
        if A1C  == "Normal":
            A1C_Result = 1
        else:
            A1C_Result = 0

    st.write("")
    st.write("")

    col1,col2,col3 = st.columns([3,4,3])
    with col2:
        button = st.button(":red[PREDICT THE READMISSION]",use_container_width= True)

        if button:
            admission = predict_readmission(Gender, Admission_Type, Diagnosis, Num_Lab_Procedures,
       Num_Medications, Num_Outpatient_Visits, Num_Inpatient_Visits,
       Num_Emergency_Visits, Num_Diagnoses, A1C_Result)
            if admission == 1:
                st.write("## :red[Readmission is Required]")
            else:
                st.write("## :green[Readmission is Not Required]")                


#__________________________________________END_____________________________________________________________
            
