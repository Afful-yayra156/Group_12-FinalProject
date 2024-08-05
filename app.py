
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd

# Load your trained model
model = tf.keras.models.load_model("C:/Users/user/OneDrive - Ashesi University/intro to ai/best_model.keras")

# Load additional information from Excel
info_df = pd.read_excel("C:/Users/user/OneDrive - Ashesi University/intro to ai/Nigerianfood_additionalinfo.xlsx")

def preprocess_image(img):
    """
    Preprocess the image for prediction.
    """
    img = img.resize((299, 299))  # Resize image to the model's expected input size
    img_array = np.array(img, dtype=np.float32)  # Convert image to numpy array with float32 type
    if img_array.ndim == 2:  # Check if image is grayscale
        img_array = np.stack([img_array] * 3, axis=-1)  # Convert grayscale to RGB
    img_array /= 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def get_food_name(predicted_class):
    """
    Retrieve the food name based on the predicted class index.
    """
    food_names = [
        'Abacha and Ugba', 'Akara and Eko', 'Amala and Gbegiri-Ewedu', 'Asaro', 'Boli(Bole)', 
        'Chin Chin', 'Egusi Soup', 'Ewa-Agoyin', 'Fried plantains(Dodo)', 'Jollof Rice', 
        'Meat Pie', 'Moin-moin', 'Nkwobi', 'Okro Soup', 'Pepper Soup', 'Puff Puff', 
        'Suya', 'Vegetable Soup'
    ]
    return food_names[predicted_class]

def get_additional_info(food_name):
    """
    Retrieve additional information from the DataFrame based on the food name.
    """
    if food_name in info_df['food_name'].values:
        info = info_df[info_df['food_name'] == food_name].iloc[0]
        return {
            'Origin or State': info['Origin_or_State'],
            'Popular Countries': info['Pop_Countries'],
            'Health Benefits': info['Health_Benefits'],
            'Calories': info['Calories'],
            'Nutrient Ratio': info['Nutrient_Ratio'],
            'Ingredients': info['Ingredients'],
            'Protein Content': info['Protein_Content'],
            'Fat Content': info['Fat_Content'],
            'Carbohydrate Content': info['Carbohydrate_Content'],
            'Allergens': info['Allergens'],
            'Mineral Content': info['Mineral-Content'],
            'Vitamin Content': info['Vitamin_Content'],
            'Suitability': info['Suitability'],
            'Fiber Content': info['Fiber_Content']
        }
    return None

def predict_and_get_info(image):
    """
    Predict the food and retrieve additional information.
    """
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make a prediction
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Retrieve additional information
    food_name = get_food_name(predicted_class)
    additional_info = get_additional_info(food_name)

    return food_name, additional_info

# Streamlit app
st.title("Nigerian Food Classifier")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Predict and get additional information
    food_name, additional_info = predict_and_get_info(image)

    # Display results
    st.write(f"Predicted Food: {food_name}")
    if additional_info:
        for key, value in additional_info.items():
            st.write(f"{key}: {value}")
    else:
        st.write("No additional information available.")
