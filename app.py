import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import io

def load_model_with_custom_objects(model_path):
    def custom_depthwise_conv2d(*args, **kwargs):
        # Remove 'groups' argument if present
        if 'groups' in kwargs:
            del kwargs['groups']
        return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)
    
    custom_objects = {
        'DepthwiseConv2D': custom_depthwise_conv2d,
        'tf': tf
    }
    
    try:
        # Attempt to load with custom objects
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Advanced loading failed: {e}")
        return None

def preprocess_image(image):
    # Resize and normalize image
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def main():
    st.title("Brain Tumor Detection Model Diagnostic")
    
    # Model path
    model_path = 'models/keras_model.h5'
    
    # Load model
    model = load_model_with_custom_objects(model_path)
    
    if model is None:
        st.error("Could not load the model. Diagnosis needed.")
        return
    
    # # Model summary
    # st.write("Model Summary:")
    # model.summary(print_fn=st.write)
    
    # # Model architecture details
    # st.write("\nModel Configuration:")
    # try:
    #     config = model.get_config()
    #     st.write(config)
    # except Exception as e:
    #     st.error(f"Could not retrieve model configuration: {e}")
    
    # Image upload for testing
    uploaded_file = st.file_uploader("Upload a test image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        try:
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            
            # Assuming 4 classes as mentioned in previous code
            CLASS_LABELS = ["Pituitary", "No Tumor", "Meningioma", "Glioma"]
            
            # Display prediction probabilities
            st.write("Prediction Probabilities:")
            for label, prob in zip(CLASS_LABELS, predictions[0]):
                st.write(f"{label}: {prob*100:.2f}%")
            
            # Predicted class
            predicted_class_index = np.argmax(predictions)
            st.write(f"\nPredicted Class: {CLASS_LABELS[predicted_class_index]}")
        
        except Exception as pred_error:
            st.error(f"Prediction failed: {pred_error}")

if __name__ == "__main__":
    main()
