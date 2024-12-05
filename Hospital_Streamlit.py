import pandas as pd
import numpy as np
import streamlit as st
import pickle
from PIL import Image
import streamlit.components.v1 as components
import base64
from datetime import datetime
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# Theme configurations
light_theme = {
    'primary_color': '#0066cc',
    'background_color': '#ffffff',
    'secondary_bg': '#f0f2f6',
    'text_color': '#262730',
    'font': 'sans-serif'
}

dark_theme = {
    'primary_color': '#4da6ff',
    'background_color': '#0e1117',
    'secondary_bg': '#262730',
    'text_color': '#fafafa',
    'font': 'sans-serif'
}

def apply_custom_css():
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stButton>button {
            width: 100%;
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.3rem;
            font-weight: 500;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            opacity: 0.85;
            transform: translateY(-2px);
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(180deg, var(--secondary-bg) 10%, var(--background-color) 90%);
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        .stSelectbox {
            margin-bottom: 1rem;
        }
        .hospital-stats {
            background-color: var(--secondary-bg);
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .stat-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.3rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def load_model():
    with open("Readmission_Model.pkl", "rb") as m:
        return pickle.load(m)

def predict_readmission(data):
    model = load_model()
    prediction = model.predict(data)
    return prediction[0]

def create_sidebar():
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=Hospital+Logo", width=150)
        st.title("Navigation")
        
        selected_page = st.radio(
            "Go to",
            ["Dashboard", "Patient Prediction", "Analytics", "Settings"]
        )
        
        st.markdown("---")
        st.subheader("Theme Settings")
        theme = st.selectbox(
            "Choose Theme",
            ["Light", "Dark"],
            key="theme_selection"
        )
        
        st.markdown("---")
        st.caption("© 2024 Hospital Management System")
        
        return selected_page, theme

def dashboard_page():
    st.title("Hospital Management Dashboard")
    
    # Mock statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", "1,234", "+12%")
    with col2:
        st.metric("Readmission Rate", "15.4%", "-2.1%")
    with col3:
        st.metric("Avg. Stay Duration", "4.2 days", "+0.3")
    
    # Mock chart
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Admissions', 'Discharges', 'Readmissions']
    )
    st.line_chart(chart_data)

def prediction_page():
    st.title("Patient Readmission Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox('Gender:', ["Female", "Male", "Other"])
        admission_type = st.selectbox('Admission Type:', ['Emergency', 'Urgent', 'Elective'])
        diagnosis = st.selectbox('Diagnosis:', ['Heart Disease', 'Diabetes', 'Injury', 'Infection'])
        lab_procedures = st.number_input('Number of Lab Procedures:', 1, 100, 1)
        medications = st.number_input('Number of Medications:', 1, 36, 1)
    
    with col2:
        outpatient_visits = st.number_input('Number of Outpatient Visits:', 0, 5, 0)
        inpatient_visits = st.number_input('Number of Inpatient Visits:', 0, 5, 0)
        emergency_visits = st.number_input('Number of Emergency Visits:', 0, 5, 0)
        num_diagnoses = st.number_input('Number of Diagnoses:', 1, 10, 1)
        a1c_result = st.selectbox('A1C Result:', ['Normal', 'Abnormal'])
    
    # Convert inputs to model format
    if st.button("Predict Readmission", key="predict_button"):
        # Convert categorical variables
        gender_code = {"Female": 0, "Male": 1, "Other": 2}[gender]
        admission_code = {"Emergency": 1, "Urgent": 2, "Elective": 0}[admission_type]
        diagnosis_code = {"Heart Disease": 1, "Diabetes": 0, "Injury": 3, "Infection": 2}[diagnosis]
        a1c_code = {"Normal": 1, "Abnormal": 0}[a1c_result]
        
        # Prepare data for prediction
        input_data = np.array([[
            gender_code, admission_code, diagnosis_code, lab_procedures,
            medications, outpatient_visits, inpatient_visits,
            emergency_visits, num_diagnoses, a1c_code
        ]])
        
        # Make prediction
        result = predict_readmission(input_data)
        
        # Display result with styling
        if result == 1:
            st.error("⚠️ High Risk: Readmission is Required")
            st.markdown("""
                ### Recommended Actions:
                1. Schedule follow-up appointment within 7 days
                2. Review medication compliance
                3. Coordinate with care management team
            """)
        else:
            st.success("✅ Low Risk: Readmission is Not Required")
            st.markdown("""
                ### Recommended Actions:
                1. Schedule routine follow-up within 30 days
                2. Provide standard discharge instructions
                3. Document any concerns for future reference
            """)

def analytics_page():
    st.title("Analytics & Insights")
    
    # Mock data for demonstration
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    readmission_data = pd.DataFrame({
        'Date': dates,
        'Readmission_Rate': np.random.uniform(10, 20, len(dates)),
        'Patient_Satisfaction': np.random.uniform(80, 95, len(dates))
    })
    
    # Plotting
    fig = px.line(readmission_data, x='Date', y=['Readmission_Rate', 'Patient_Satisfaction'],
                  title='Hospital Performance Metrics')
    st.plotly_chart(fig)

def settings_page():
    st.title("Settings")
    
    st.subheader("User Preferences")
    st.checkbox("Enable email notifications")
    st.checkbox("Enable SMS alerts")
    
    st.subheader("System Settings")
    st.selectbox("Default prediction threshold", ["Low", "Medium", "High"])
    st.selectbox("Data refresh frequency", ["Hourly", "Daily", "Weekly"])

def main():
    st.set_page_config(
        page_title="Hospital Readmission Prediction System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()
    selected_page, theme = create_sidebar()
    
    # Apply selected theme
    current_theme = dark_theme if theme == "Dark" else light_theme
    st.markdown(f"""
        <style>
        :root {{
            --primary-color: {current_theme['primary_color']};
            --background-color: {current_theme['background_color']};
            --secondary-bg: {current_theme['secondary_bg']};
            --text-color: {current_theme['text_color']};
            --font: {current_theme['font']};
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # Display selected page
    if selected_page == "Dashboard":
        dashboard_page()
    elif selected_page == "Patient Prediction":
        prediction_page()
    elif selected_page == "Analytics":
        analytics_page()
    elif selected_page == "Settings":
        settings_page()

if __name__ == "__main__":
    main()
