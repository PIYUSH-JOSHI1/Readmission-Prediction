import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import pickle
import os
from io import BytesIO
import pydicom
import cv2
import random
from ultralytics import YOLO
from plotly.subplots import make_subplots

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

# 3D Scatter plot for visual appeal
def create_3d_scatter():
    x = np.random.randn(100)
    y = np.random.randn(100)
    z = np.random.randn(100)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=z,
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(r=0, b=0, l=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_dynamic_dashboard():
    st.title("Hospital Dashboard")
    
    st_autorefresh(interval=10000, key="dashboard_refresh")
    
    current_time = datetime.now()
    times = pd.date_range(end=current_time, periods=20, freq='1min')
    
    # 3D Scatter plot for visual appeal
    st.subheader("Hospital Activity Visualization")
    fig_3d = create_3d_scatter()
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Interactive metrics with hover effect
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
    
    # Interactive charts
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Patient Flow (Last 20 minutes)", "Department Load (%)"))
    
    # Patient Flow
    fig.add_trace(
        go.Scatter(
            x=times,
            y=np.random.randint(50, 100, size=20),
            name="Admissions",
            mode='lines+markers',
            line=dict(color='#4da6ff', width=3),
            marker=dict(size=8, symbol='circle', line=dict(color='#ffffff', width=2))
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=np.random.randint(40, 90, size=20),
            name="Discharges",
            mode='lines+markers',
            line=dict(color='#ff4961', width=3),
            marker=dict(size=8, symbol='circle', line=dict(color='#ffffff', width=2))
        ),
        row=1, col=1
    )
    
    # Department Load
    departments = ['ER', 'ICU', 'Surgery', 'Pediatrics', 'General']
    values = np.random.randint(40, 100, size=len(departments))
    fig.add_trace(
        go.Bar(
            x=departments,
            y=values,
            marker_color='#4da6ff',
            hoverinfo='y',
            textposition='auto',
            textfont=dict(color='white'),
            hoverlabel=dict(bgcolor='#1a1a1a', font_size=14)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa'),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    emergency_data = pd.DataFrame({
        'Time': times[-5:],
        'Type': np.random.choice(['Critical', 'Moderate', 'Minor'], size=5),
        'Department': np.random.choice(['ER', 'ICU', 'Surgery'], size=5),
        'Status': np.random.choice(['In Progress', 'Waiting', 'Completed'], size=5)
    })
    
    st.subheader("Recent Emergency Cases")
    st.dataframe(
        emergency_data.style
        .applymap(lambda x: f'color: {dark_theme["text_color"]}; background-color: {dark_theme["card_bg"]}')
        .set_properties(**{'background-color': dark_theme['card_bg'], 'color': dark_theme['text_color'], 'border-color': dark_theme['primary_color']})
        .highlight_max(axis=0, props='color: #ff4961; font-weight: bold;')
        .highlight_min(axis=0, props='color: #2fd36e; font-weight: bold;'),
        use_container_width=True
    )

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
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Admissions vs Discharges Trend", "Readmission Trend", "Average Length of Stay Trend", "Department-wise Statistics"))
    
    # Admissions vs Discharges Trend
    fig.add_trace(go.Scatter(
        x=analytics_data['Date'],
        y=analytics_data['Admissions'],
        name="Admissions",
        line=dict(color='#4da6ff', width=3)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=analytics_data['Date'],
        y=analytics_data['Discharges'],
        name="Discharges",
        line=dict(color='#ff4961', width=3)
    ), row=1, col=1)
    
    # Readmission Trend
    fig.add_trace(go.Scatter(
        x=analytics_data['Date'],
        y=analytics_data['Readmissions'],
        name="Readmissions",
        line=dict(color='#ffd534', width=3)
    ), row=1, col=2)
    
    # Average Length of Stay Trend
    fig.add_trace(go.Scatter(
        x=analytics_data['Date'],
        y=analytics_data['Average_Stay'],
        name="Average Stay",
        line=dict(color='#2fd36e', width=3)
    ), row=2, col=1)
    
    # Department-wise Statistics
    departments = ['Emergency', 'Surgery', 'Cardiology', 'Pediatrics', 'Neurology']
    occupancy_rates = np.random.uniform(60, 95, len(departments))
    fig.add_trace(go.Bar(
        x=departments,
        y=occupancy_rates,
        name="Occupancy Rate",
        marker_color='#4da6ff'
    ), row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa'),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        )
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Department-wise Statistics")
    dept_data = pd.DataFrame({
        'Department': departments,
        'Occupancy_Rate': occupancy_rates,
        'Avg_Stay': np.random.uniform(2, 8, len(departments)),
        'Patient_Satisfaction': np.random.uniform(75, 95, len(departments))
    })
    
    st.dataframe(
        dept_data.style.format({
            'Occupancy_Rate': '{:.2f}%',
            'Avg_Stay': '{:.2f}',
            'Patient_Satisfaction': '{:.2f}%'
        })
        .applymap(lambda x: f'color: {dark_theme["text_color"]}; background-color: {dark_theme["card_bg"]}')
        .set_properties(**{'background-color': dark_theme['card_bg'], 'color': dark_theme['text_color'], 'border-color': dark_theme['primary_color']})
        .highlight_max(axis=0, props='color: #2fd36e; font-weight: bold;')
        .highlight_min(axis=0, props='color: #ff4961; font-weight: bold;'),
        use_container_width=True
    )

def medical_image_analysis_page():
    st.title("Medical Image Analysis")
    
    # Disease details dictionary
    DISEASE_DETAILS = {
        'Aortic enlargement': {
            'description': 'Abnormal enlargement of the aorta',
            'precautions': [
                'Immediate cardiovascular consultation',
                'Regular blood pressure monitoring',
                'Avoid heavy lifting and strenuous activities'
            ],
            'admission': 'Immediate hospitalization if risk of rupture'
        },
        'Cardiomegaly': {
            'description': 'Abnormal enlargement of the heart',
            'precautions': [
                'Restrict physical activities',
                'Follow strict medication regimen',
                'Regular cardiac monitoring'
            ],
            'admission': 'Urgent cardiac care if symptoms are severe'
        },
        'Pneumothorax': {
            'description': 'Collapsed or partially collapsed lung',
            'precautions': [
                'Oxygen therapy',
                'Chest tube insertion may be required',
                'Complete bed rest'
            ],
            'admission': 'Immediate hospitalization'
        },
        'No finding': {
            'description': 'No significant medical conditions detected',
            'precautions': ['Regular health check-ups'],
            'admission': 'Not required'
        }
    }
    
    # Ensure uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    # Load YOLO model
    MODEL_PATH = 'yolov8n.pt'
    model = YOLO(MODEL_PATH)
    
    # Disease classes
    CLASSES = [
        'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
        'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
        'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
        'Pulmonary fibrosis', 'No finding'
    ]
    
    st.markdown("### Upload Medical Image")
    uploaded_file = st.file_uploader(
        "Choose a medical image (DICOM, JPG, PNG)", 
        type=['dcm', 'dicom', 'jpg', 'png']
    )
    
    if uploaded_file is not None:
        # Save the uploaded file
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_path = os.path.join("uploads", uploaded_file.name)
        
        try:
            # Read the image
            if uploaded_file.name.endswith('.dcm') or uploaded_file.name.endswith('.dicom'):
                dicom = pydicom.dcmread(file_path)
                img = dicom.pixel_array
            else:
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize and convert to RGB
            img = (img / img.max() * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Perform inference
            results = model(img)
            
            # Process results
            predictions = []
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                
                for box, confidence, class_id in zip(boxes, confidences, class_ids):
                    x_min, y_min, x_max, y_max = box
                    predictions.append({
                        'class_name': CLASSES[class_id],
                        'confidence': float(confidence),
                        'box': {
                            'x_min': int(x_min),
                            'y_min': int(y_min),
                            'x_max': int(x_max),
                            'y_max': int(y_max)
                        }
                    })
            
            # If no findings, add a default prediction
            if not predictions:
                predictions.append({
                    'class_name': 'No finding',
                    'confidence': 1.0,
                    'box': {'x_min': 0, 'y_min': 0, 'x_max': 1, 'y_max': 1}
                })
            
            # Visualize results
            visualized_img = img.copy()
            for pred in predictions:
                color = [random.randint(0, 255) for _ in range(3)]
                visualized_img = draw_bbox(visualized_img, 
                                           [pred['box']['x_min'], pred['box']['y_min'], 
                                            pred['box']['x_max'], pred['box']['y_max']],
                                           pred['class_name'], 
                                           pred['confidence'],
                                           color)
            
            st.subheader("Analysis Results")
            
            # Create responsive layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display image responsively
                st.image(visualized_img, use_column_width=True, caption="Analyzed Medical Image")
            
            with col2:
                # Detailed analysis report
                st.markdown("### Detected Conditions")
                
                # Check if predictions exist
                if not predictions:
                    st.info("No significant findings detected.")
                else:
                    for pred in predictions:
                        disease = pred['class_name']
                        confidence = pred['confidence']
                        
                        st.markdown(f"#### {disease}")
                        st.markdown(f"**Confidence:** {confidence:.2%}")
                        
                        # Retrieve disease details
                        if disease in DISEASE_DETAILS:
                            details = DISEASE_DETAILS[disease]
                            st.markdown(f"**Description:** {details['description']}")
                            
                            st.markdown("**Precautions:**")
                            for precaution in details['precautions']:
                                st.markdown(f"- {precaution}")
                            
                            st.markdown(f"**Hospital Admission:** {details['admission']}")
                        
                        st.markdown("---")
            
            # Clean up temporary file
            os.remove(file_path)
        
        except Exception as e:
            st.error(f"An error occurred during image analysis: {str(e)}")

def draw_bbox(image, box, label, confidence, color):   
    alpha = 0.1
    alpha_box = 0.4
    overlay_bbox = image.copy()
    overlay_text = image.copy()
    output = image.copy()

    text_width, text_height = cv2.getTextSize(label.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.rectangle(overlay_bbox, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, -1)
    cv2.addWeighted(overlay_bbox, alpha, output, 1 - alpha, 0, output)
    cv2.rectangle(overlay_text, (int(box[0]), int(box[1])-7-text_height), (int(box[0])+text_width+2, int(box[1])), (0, 0, 0), -1)
    cv2.addWeighted(overlay_text, alpha_box, output, 1 - alpha_box, 0, output)
    cv2.rectangle(output, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    cv2.putText(output, label.upper(), (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, f"{confidence:.2f}", (int(box[0]), int(box[3])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return output

import streamlit as st
from transformers import pipeline
import requests

def chatbot_page():
    st.title("Hospital Assistant Chatbot")
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    st.write("Chat with our hospital assistant for help and information.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the hospital..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get chatbot response using HuggingFace API
        API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
        headers = {"Authorization": "Bearer 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"}  # Get free API key from huggingface.co
        
        try:
            # Make API call
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
            response_data = response.json()
            
            # Extract and clean bot response
            bot_response = response_data[0]["generated_text"]
            
            # Add bot response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            
            # Display bot response
            with st.chat_message("assistant"):
                st.write(bot_response)
                
        except Exception as e:
            st.error(f"Error getting response from chatbot: {str(e)}")
            
    # Add clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# Add chatbot to main navigation
import tensorflow as tf
from PIL import Image

# Add this function after the medical_image_analysis_page() function
def brain_tumor_detection_page():
    st.title("Brain Tumor Detection")
    
    def load_model_with_custom_objects(model_path):
        def custom_depthwise_conv2d(*args, **kwargs):
            if 'groups' in kwargs:
                del kwargs['groups']
            return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)
        
        custom_objects = {
            'DepthwiseConv2D': custom_depthwise_conv2d,
            'tf': tf
        }
        
        try:
            model = tf.keras.models.load_model(
                model_path, 
                custom_objects=custom_objects,
                compile=False
            )
            return model
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            return None

    def preprocess_image(image):
        image = image.convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    model_path = 'models/keras_model.h5'
    model = load_model_with_custom_objects(model_path)
    
    if model is None:
        st.error("Could not load the model.")
        return
    
    uploaded_file = st.file_uploader("Upload brain scan image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan", use_column_width=True)
        
        try:
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            
            CLASS_LABELS = ["Pituitary", "No Tumor", "Meningioma", "Glioma"]
            
            st.subheader("Prediction Results:")
            for label, prob in zip(CLASS_LABELS, predictions[0]):
                st.progress(float(prob))
                st.write(f"{label}: {prob*100:.2f}%")
            
            predicted_class = CLASS_LABELS[np.argmax(predictions)]
            st.success(f"Predicted Condition: {predicted_class}")
            
        except Exception as e:
            st.error(f"Analysis failed: {e}")


def main():
    st.set_page_config(
        page_title="Hospital Management System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if 'language' not in st.session_state:
        st.session_state.language = 'English'
    
    languages = load_languages()
    
    # Sidebar Navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=Hospital+Logo", width=150)
        
        if 'user_profile' in st.session_state and st.session_state.user_profile['name']:
            st.write(f"Welcome, {st.session_state.user_profile['name']}")
            if st.session_state.user_profile['profile_picture']:
                st.image(st.session_state.user_profile['profile_picture'], width=100)
        
        # Navigation menu
        menu_options = [
            "Dashboard", 
            "Patient Prediction", 
            "Analytics", 
            "Brain Tumor Detection",
            "Medical Image Analysis",
            "User Profile", 
            "Emergency Contact", 
            "About Us", 
            "Settings",
            "Chatbot" 
        ]
        
        selected_page = st.selectbox(
            languages[st.session_state.language]['welcome'],
            menu_options
        )
    
     
    if selected_page == "Chatbot":
        chatbot_page()
    # Page rendering based on selection
    if selected_page == "Dashboard":
        create_dynamic_dashboard()
    elif selected_page == "Patient Prediction":
        prediction_page()
    elif selected_page == "Brain Tumor Detection":
        brain_tumor_detection_page()
    elif selected_page == "Analytics":
        analytics_page()
    elif selected_page == "Medical Image Analysis":
        medical_image_analysis_page()
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
            ["Dark"]
        )
        
        current_theme = dark_theme
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

    # Add 3D effect and continuous movement
    st.markdown("""
    <script>
    const cursor = document.createElement('div');
    cursor.className = 'custom-cursor';
    document.body.appendChild(cursor);

    document.addEventListener('mousemove', (e) => {
        cursor.style.left = e.clientX + 'px';
        cursor.style.top = e.clientY + 'px';
        
        const elements = document.querySelectorAll('.stPlotlyChart, .stDataFrame, .stMetric');
        elements.forEach(el => {
            const rect = el.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width - 0.5;
            const y = (e.clientY - rect.top) / rect.height - 0.5;
            el.style.transform = `perspective(1000px) rotateY(${x * 5}deg) rotateX(${-y * 5}deg)`;
        });
    });
    </script>
    <style>
    .custom-cursor {
        width: 20px;
        height: 20px;
        border: 2px solid #ffffff;
        border-radius: 50%;
        position: fixed;
        pointer-events: none;
        z-index: 9999;
        mix-blend-mode: difference;
    }
    .stPlotlyChart, .stDataFrame, .stMetric {
        transition: transform 0.1s ease;
    }
    .stMetric {
        background: linear-gradient(145deg, rgba(26,26,26,0.6) 0%, rgba(26,26,26,0.8) 100%);
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric:hover {
        background: linear-gradient(145deg, rgba(26,26,26,0.8) 0%, rgba(26,26,26,1) 100%);
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
