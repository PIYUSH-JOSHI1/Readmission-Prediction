import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import pickle
import os

# Load languages
def load_languages():
    return {
        'English': {
            'welcome': 'Welcome',
            'dashboard': 'Dashboard',
            'profile': 'Profile',
            'settings': 'Settings',
            'emergency': 'Emergency Contact',
            'about': 'About Us',
            'prediction': 'Patient Prediction',
            'analytics': 'Analytics'
        },
        'Spanish': {
            'welcome': 'Bienvenido',
            'dashboard': 'Tablero',
            'profile': 'Perfil',
            'settings': 'Ajustes',
            'emergency': 'Contacto de Emergencia',
            'about': 'Sobre Nosotros',
            'prediction': 'Predicción de Pacientes',
            'analytics': 'Análisis'
        },
        'French': {
            'welcome': 'Bienvenue',
            'dashboard': 'Tableau de Bord',
            'profile': 'Profil',
            'settings': 'Paramètres',
            'emergency': 'Contact d\'urgence',
            'about': 'À Propos',
            'prediction': 'Prédiction de Patients',
            'analytics': 'Analytique'
        }
    }

# Theme configurations
light_theme = {
    'primary_color': '#0066cc',
    'background_color': '#ffffff',
    'secondary_bg': '#f0f2f6',
    'text_color': '#262730',
    'font': 'sans-serif',
    'card_bg': '#ffffff',
    'success_color': '#28a745',
    'warning_color': '#ffc107',
    'danger_color': '#dc3545'
}

dark_theme = {
    'primary_color': '#4da6ff',
    'background_color': '#0e1117',
    'secondary_bg': '#262730',
    'text_color': '#fafafa',
    'font': 'sans-serif',
    'card_bg': '#1a1a1a',
    'success_color': '#2fd36e',
    'warning_color': '#ffd534',
    'danger_color': '#ff4961'
}

def user_profile_section():
    st.title("User Profile")
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'email': '',
            'phone': '',
            'department': '',
            'role': '',
            'profile_picture': None
        }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.user_profile['name'] = st.text_input("Full Name", st.session_state.user_profile['name'])
        st.session_state.user_profile['email'] = st.text_input("Email", st.session_state.user_profile['email'])
        st.session_state.user_profile['phone'] = st.text_input("Phone", st.session_state.user_profile['phone'])
        
    with col2:
        st.session_state.user_profile['department'] = st.selectbox(
            "Department",
            ["Cardiology", "Emergency", "Pediatrics", "Surgery", "Other"],
            index=0 if not st.session_state.user_profile['department'] else None
        )
        st.session_state.user_profile['role'] = st.selectbox(
            "Role",
            ["Doctor", "Nurse", "Administrator", "Other"],
            index=0 if not st.session_state.user_profile['role'] else None
        )
        uploaded_file = st.file_uploader("Upload Profile Picture", type=['jpg', 'png'])
        if uploaded_file is not None:
            st.session_state.user_profile['profile_picture'] = uploaded_file
            st.image(uploaded_file, width=150)
    
    if st.button("Save Profile"):
        st.success("Profile updated successfully!")

def emergency_contact_section():
    st.title("Emergency Contact")
    
    st.header("Emergency Department")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📞 Emergency Hotline\n\n1-800-HOSPITAL")
    with col2:
        st.info("🚑 Ambulance Service\n\n911")
    with col3:
        st.info("👨‍⚕️ On-call Doctor\n\n+1-555-0123")
    
    st.header("Contact Form")
    
    emergency_type = st.selectbox(
        "Emergency Type",
        ["Medical Emergency", "Fire Emergency", "Security Issue", "Other"]
    )
    
    description = st.text_area("Description of Emergency")
    location = st.text_input("Location in Hospital")
    
    if st.button("Submit Emergency Alert"):
        if description and location:
            st.success("Emergency alert submitted successfully!")
            st.info("Emergency response team has been notified.")
        else:
            st.error("Please fill in all required fields.")

def about_us_section():
    st.title("About Us")
    
    st.markdown("""
    ## Our Mission
    To provide exceptional healthcare services with compassion and innovation, 
    ensuring the best possible outcomes for our patients and communities.
    
    ## Our Vision
    To be the leading healthcare provider known for excellence in patient care, 
    medical research, and healthcare technology innovation.
    
    ## Our Values
    - **Excellence** in all aspects of healthcare
    - **Compassion** towards all patients
    - **Innovation** in medical practices
    - **Integrity** in our actions
    - **Teamwork** in our approach
    
    ## Hospital Statistics
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Years of Service", "50+")
    with col2:
        st.metric("Healthcare Professionals", "1000+")
    with col3:
        st.metric("Patients Served Annually", "50,000+")
    
    st.header("Our Departments")
    departments = [
        ("🫀 Cardiology", "Specialized heart care and treatment"),
        ("🧠 Neurology", "Expert neurological care"),
        ("👶 Pediatrics", "Comprehensive children's healthcare"),
        ("🏥 Emergency Care", "24/7 emergency services"),
        ("🔬 Research", "Cutting-edge medical research")
    ]
    
    for dept, desc in departments:
        st.subheader(dept)
        st.write(desc)

def create_dynamic_dashboard():
    st.title("Hospital Dashboard")
    
    st_autorefresh(interval=10000, key="dashboard_refresh")
    
    current_time = datetime.now()
    times = pd.date_range(end=current_time, periods=20, freq='1min')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        current_patients = np.random.randint(80, 120)
        st.metric("Current Patients", current_patients, delta=np.random.randint(-5, 5))
    with col2:
        bed_capacity = f"{np.random.randint(60, 90)}%"
        st.metric("Bed Capacity", bed_capacity, delta=f"{np.random.randint(-3, 3)}%")
    with col3:
        staff_on_duty = np.random.randint(40, 60)
        st.metric("Staff on Duty", staff_on_duty, delta=np.random.randint(-2, 2))
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=times,
            y=np.random.randint(50, 100, size=20),
            name="Admissions",
            mode='lines+markers'
        ))
        fig1.add_trace(go.Scatter(
            x=times,
            y=np.random.randint(40, 90, size=20),
            name="Discharges",
            mode='lines+markers'
        ))
        fig1.update_layout(title="Patient Flow (Last 20 minutes)")
        st.plotly_chart(fig1)
    
    with col2:
        departments = ['ER', 'ICU', 'Surgery', 'Pediatrics', 'General']
        values = np.random.randint(40, 100, size=len(departments))
        fig2 = go.Figure(data=[go.Bar(x=departments, y=values)])
        fig2.update_layout(title="Department Load (%)")
        st.plotly_chart(fig2)
    
    emergency_data = pd.DataFrame({
        'Time': times[-5:],
        'Type': np.random.choice(['Critical', 'Moderate', 'Minor'], size=5),
        'Department': np.random.choice(['ER', 'ICU', 'Surgery'], size=5),
        'Status': np.random.choice(['In Progress', 'Waiting', 'Completed'], size=5)
    })
    
    st.subheader("Recent Emergency Cases")
    st.dataframe(emergency_data, use_container_width=True)

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
    
    if st.button("Predict Readmission", key="predict_button"):
        gender_code = {"Female": 0, "Male": 1, "Other": 2}[gender]
        admission_code = {"Emergency": 1, "Urgent": 2, "Elective": 0}[admission_type]
        diagnosis_code = {"Heart Disease": 1, "Diabetes": 0, "Injury": 3, "Infection": 2}[diagnosis]
        a1c_code = {"Normal": 1, "Abnormal": 0}[a1c_result]
        
        input_data = np.array([[
            gender_code, admission_code, diagnosis_code, lab_procedures,
            medications, outpatient_visits, inpatient_visits,
            emergency_visits, num_diagnoses, a1c_code
        ]])
        
        try:
            model_path = os.path.join(os.path.dirname(__file__), "Readmission_Model.pkl")
            with open(model_path, "rb") as m:
                model = pickle.load(m)
            result = model.predict(input_data)[0]
            
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
        except FileNotFoundError:
            st.error("Model file not found. Please ensure 'Readmission_Model.pkl' is in the correct directory.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

def analytics_page():
    st.title("Hospital Analytics Dashboard")
    
    time_period = st.selectbox(
        "Select Time Period",
        ["Last 24 Hours", "Last Week", "Last Month", "Last Year"]
    )
    
    if time_period == "Last 24 Hours":
        dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
    elif time_period == "Last Week":
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
    elif time_period == "Last Month":
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    else:
        dates = pd.date_range(end=datetime.now(), periods=12, freq='M')
    
    analytics_data = pd.DataFrame({
        'Date': dates,
        'Admissions': np.random.randint(50, 150, size=len(dates)),
        'Discharges': np.random.randint(40, 140, size=len(dates)),
        'Readmissions': np.random.randint(5, 30, size=len(dates)),
        'Average_Stay': np.random.uniform(2, 7, size=len(dates))
    })
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_admissions = analytics_data['Admissions'].sum()
        st.metric("Total Admissions", f"{total_admissions:,}")
    
    with col2:
        avg_stay = analytics_data['Average_Stay'].mean()
        st.metric("Average Stay (days)", f"{avg_stay:.1f}")
    
    with col3:
        readmission_rate = (analytics_data['Readmissions'].sum() / total_admissions) * 100
        st.metric("Readmission Rate", f"{readmission_rate:.1f}%")
    
    with col4:
        bed_turnover = total_admissions / len(dates)
        st.metric("Daily Bed Turnover", f"{bed_turnover:.1f}")
    
    st.subheader("Admissions vs Discharges Trend")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=analytics_data['Date'],
        y=analytics_data['Admissions'],
        name="Admissions",
        line=dict(color='blue')
    ))
    fig1.add_trace(go.Scatter(
        x=analytics_data['Date'],
        y=analytics_data['Discharges'],
        name="Discharges",
        line=dict(color='green')
    ))
    fig1.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Patients",
        hovermode='x unified'
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Readmission Trend")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=analytics_data['Date'],
            y=analytics_data['Readmissions'],
            line=dict(color='red')
        ))
        fig2.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Readmissions",
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("Average Length of Stay Trend")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=analytics_data['Date'],
            y=analytics_data['Average_Stay'],
            line=dict(color='purple')
        ))
        fig3.update_layout(
            xaxis_title="Date",
            yaxis_title="Days",
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("Department-wise Statistics")
    departments = ['Emergency', 'Surgery', 'Cardiology', 'Pediatrics', 'Neurology']
    dept_data = pd.DataFrame({
        'Department': departments,
        'Occupancy_Rate': np.random.uniform(60, 95, len(departments)),
        'Avg_Stay': np.random.uniform(2, 8, len(departments)),
        'Patient_Satisfaction': np.random.uniform(75, 95, len(departments))
    })
    
    st.dataframe(dept_data.round(2), use_container_width=True)

def main():
    st.set_page_config(
        page_title="Hospital Management System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if 'language' not in st.session_state:
        st.session_state.language = 'English'
    
    languages = load_languages()
    
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=Hospital+Logo", width=150)
        
        if 'user_profile' in st.session_state and st.session_state.user_profile['name']:
            st.write(f"Welcome, {st.session_state.user_profile['name']}")
            if st.session_state.user_profile['profile_picture']:
                st.image(st.session_state.user_profile['profile_picture'], width=100)
        
        selected_page = st.radio(
            languages[st.session_state.language]['welcome'],
            ["Dashboard", "Patient Prediction", "Analytics", "User Profile", 
             "Emergency Contact", "About Us", "Settings"]
        )
    
    if selected_page == "Dashboard":
        create_dynamic_dashboard()
    elif selected_page == "Patient Prediction":
        prediction_page()
    elif selected_page == "Analytics":
        analytics_page()
    elif selected_page == "User Profile":
        user_profile_section()
    elif selected_page == "Emergency Contact":
        emergency_contact_section()
    elif selected_page == "About Us":
        about_us_section()
    elif selected_page == "Settings":
        st.title("Settings")
        
        st.subheader("Language Settings")
        new_language = st.selectbox(
            "Select Language",
            list(languages.keys()),
            index=list(languages.keys()).index(st.session_state.language)
        )
        if new_language != st.session_state.language:
            st.session_state.language = new_language
            st.experimental_rerun()
        
        st.subheader("Theme Settings")
        theme = st.selectbox(
            "Choose Theme",
            ["Light", "Dark"]
        )
        
        current_theme = dark_theme if theme == "Dark" else light_theme
        st.markdown(f"""
            <style>
            :root {{
                --primary-color: {current_theme['primary_color']};
                --background-color: {current_theme['background_color']};
                --secondary-bg: {current_theme['secondary_bg']};
                --text-color: {current_theme['text_color']};
                --font: {current_theme['font']};
                --card-bg: {current_theme['card_bg']};
                --success-color: {current_theme['success_color']};
                --warning-color: {current_theme['warning_color']};
                --danger-color: {current_theme['danger_color']};
            }}
            </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
