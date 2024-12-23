# 🏥 Hospital Management System

## 📋 Overview
A comprehensive hospital management system built with Streamlit, featuring real-time analytics, medical image analysis, patient prediction, and multilingual support. The system includes advanced features like brain tumor detection, chatbot assistance, and dynamic dashboards.

## ⭐ Features

### 🔷 Core Functionalities
- **📊 Dynamic Dashboard**
  - Real-time patient flow monitoring
  - Interactive 3D data visualization
  - Department-wise statistics
  - Bed capacity tracking
  - Staff monitoring

### 🔬 Medical Analysis Tools
- **🧠 Brain Tumor Detection**
  - AI-powered tumor classification
  - Support for multiple tumor types (Pituitary, Meningioma, Glioma)
  - Real-time image processing
  - Confidence score display

- **🔍 Medical Image Analysis**
  - DICOM file support
  - Multiple format support (JPG, PNG)
  - Disease detection and classification
  - Automated reporting system
  - Visual annotations with confidence scores

### 👥 Patient Management
- **📈 Patient Prediction System**
  - Readmission risk analysis
  - Multiple factor consideration
  - Automated recommendations
  - Risk factor visualization

- **📊 Analytics Dashboard**
  - Patient flow trends
  - Department-wise statistics
  - Length of stay analysis
  - Interactive visualizations
  - Custom time period selection

### 🤝 Support Features
- **🤖 Hospital Assistant Chatbot**
  - Natural language processing
  - PDF/TXT export functionality
  - Chat history management
  - Real-time responses

- **🚨 Emergency Contact System**
  - Quick access to emergency services
  - Emergency alert submission
  - Location tracking
  - Priority-based routing

## 💻 Technical Requirements

### 📦 Dependencies
```python
streamlit
pandas
numpy
plotly
opencv-python
tensorflow
pillow
pydicom
google-cloud-aiplatform
ultralytics
fpdf
```

### ⚙️ Additional Requirements
- Python 3.8+
- CUDA-compatible GPU (for AI models)
- Minimum 8GB RAM
- 50GB storage space

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/PIYUSH-JOSHI1/Readmission-Prediction.git
cd Readmission-Prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
export API_KEY="your-api-key"
```

4. Run the application:
```bash
streamlit run main.py
```

## ⚙️ Configuration

### 🌐 Language Settings
The system supports multiple languages:
- 🇺🇸 English (default)
- 🇪🇸 Spanish
- 🇫🇷 French

Configure language settings in the settings menu.

### 🎨 Theme Configuration
Currently supports:
- 🌙 Dark theme (default)
Custom themes can be configured in `dark_theme` dictionary.

## 🔒 Security Features
- Secure file handling
- API key protection
- Session state management
- Secure data transmission

## 🤖 Model Information

### 🧠 Brain Tumor Detection Model
- Architecture: Custom CNN
- Input size: 224x224x3
- Output classes: 4 (Pituitary, No Tumor, Meningioma, Glioma)

### 🔍 Medical Image Analysis Model
- Framework: YOLO v8
- Supported formats: DICOM, JPG, PNG
- Real-time detection capabilities

## 👥 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## 🙏 Acknowledgments
- TensorFlow team for the deep learning framework
- Streamlit team for the web framework
- YOLO team for the object detection model
- Google Cloud team for the AI Platform services

## 💬 Support
For support, email: support@hospitalsystem.com or raise an issue in the repository.

## 🗺️ Roadmap
- [ ] Integration with Electronic Health Records
- [ ] Mobile application development
- [ ] Additional language support
- [ ] Advanced analytics features
- [ ] Real-time patient monitoring
- [ ] Integration with medical devices

## 🏗️ System Architecture
```
hospital-management-system/
├── main.py
├── models/
│   ├── keras_model.h5
│   └── yolov8n.pt
├── uploads/
├── static/
│   └── assets/
├── utils/
│   ├── image_processing.py
│   └── data_analysis.py
└── config/
    └── settings.py
```
