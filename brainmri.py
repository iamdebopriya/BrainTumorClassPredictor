import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
model = load_model('model (1).keras')

# Class mappings
class_mappings = {0: 'Glioma', 1: 'Meninigioma', 2: 'Notumor', 3: 'Pituitary'}

# Function to load and preprocess the image
def preprocess_image(img):
    img = img.resize((168, 168))
    img = img.convert('L')  # convert to grayscale
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Detailed advice dictionary
advice = {
    'Glioma': {
        'description': 'Gliomas are  type of tumor that occurs in the brain and spinal cord. They begin in the glial cells that surround nerve cells and help them function.',
        'location': 'Brain and spinal cord.',
        'precautions': 'Regular monitoring with MRI scans, avoiding radiation exposure, and maintaining a healthy lifestyle.',
        'medications': 'Chemotherapy drugs like temozolomide, corticosteroids to reduce inflammation, and anticonvulsants to prevent seizures.',
        'doctors': 'Neuro-oncologist, neurosurgeon, radiologist.',
        'surgery': 'Surgical removal of the tumor, often followed by radiation therapy and/or chemotherapy.',
        'diet': 'High-fiber diet with plenty of fruits and vegetables, lean proteins, and whole grains. Avoid processed foods and high-sugar diets.'
    },
    'Meninigioma': {
        'description': 'Meningiomas are tumors that arise from the meninges, the membranes that surround your brain and spinal cord. Most are benign, but they can cause significant problems due to their size and location.',
        'location': 'Membranes covering the brain and spinal cord.',
        'precautions': 'Regular MRI or CT scans, managing symptoms, and maintaining a healthy diet and exercise routine.',
        'medications': 'Anti-seizure medications, corticosteroids to reduce swelling, and pain relievers.',
        'doctors': 'Neurosurgeon, neurologist, oncologist.',
        'surgery': 'Surgical resection is often the first line of treatment. Radiation therapy may follow if the tumor cannot be completely removed.',
        'diet': 'Balanced diet with a focus on anti-inflammatory foods like leafy greens, nuts, and fatty fish. Limit red meat and sugary foods.'
    },
    'Notumor': {
        'description': 'No tumor detected.',
        'location': 'N/A',
        'precautions': 'Maintain regular health check-ups and MRI scans as advised by your doctor.',
        'medications': 'N/A',
        'doctors': 'Primary care physician, neurologist (if needed).',
        'surgery': 'N/A',
        'diet': 'Maintain a healthy and balanced diet rich in fruits, vegetables, lean proteins, and whole grains.'
    },
    'Pituitary': {
        'description': 'Pituitary tumors are abnormal growths that develop in your pituitary gland, which can affect hormone levels and overall health.',
        'location': 'Pituitary gland, located at the base of the brain.',
        'precautions': 'Regular MRI scans, hormone level monitoring, and managing symptoms.',
        'medications': 'Hormone replacement therapy, medications to shrink the tumor, and corticosteroids.',
        'doctors': 'Endocrinologist, neurosurgeon, neurologist.',
        'surgery': 'Transsphenoidal surgery to remove the tumor, radiation therapy if surgery is not completely successful.',
        'diet': 'Balanced diet with adequate calcium and vitamin D, especially if hormone levels are affected. Avoid caffeine and alcohol.'
    }
}

# Function to get predictions and advice
def get_prediction_and_advice(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_mappings[predicted_class_index]
    detailed_advice = advice[predicted_class]
    
    return predicted_class, predictions[0], detailed_advice

# Apply custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #a2c2e8, #f4f4f9); /* Light blue to light gray gradient */
    }
    .stTitle {
        color: #007bff; /* Title color */
        font-size: 2em; /* Title size */
    }
    .stButton > button {
        background-color: #007bff; /* Button background color */
        color: #fff; /* Button text color */
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #0056b3; /* Button hover color */
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("Brain Tumor MRI Classification")
st.write("Upload an MRI image and get predictions along with professional advice.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI.', use_column_width=True)
    
    st.write("")
    st.write("Classifying...")
    
    predicted_class, probabilities, detailed_advice = get_prediction_and_advice(img)
    
    st.write(f"Prediction: **{predicted_class.capitalize()}**")
    st.write(f"**Description:** {detailed_advice['description']}")
    st.write(f"**Location:** {detailed_advice['location']}")
    st.write(f"**Precautions:** {detailed_advice['precautions']}")
    st.write(f"**Medications:** {detailed_advice['medications']}")
    st.write(f"**Doctors:** {detailed_advice['doctors']}")
    st.write(f"**Surgery:** {detailed_advice['surgery']}")
    st.write(f"**Diet:** {detailed_advice['diet']}")
    
    # Bar plot for prediction probabilities
    fig, ax = plt.subplots()
    ax.barh([class_mappings[i] for i in range(4)], probabilities)
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Probabilities')
    st.pyplot(fig)
    
