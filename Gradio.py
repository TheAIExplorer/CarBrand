import os
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import cv2

# Load the best model
model_path = "car_classification_model.h5"
best_model = load_model(model_path)

class_labels = ['Audi', 'Hyundai Creta', 'Mahindra Scorpio', 'Rolls Royce', 'Swift', 'Tata Safari', 'Toyota Innova']

# Define a tf.function for prediction
@tf.function
def predict_image(image_array):
    prediction = best_model(image_array)
    class_index = tf.argmax(prediction, axis=1)
    predicted_class = tf.gather(class_labels, class_index)
    return predicted_class

# Predict function with breed name as string
def Predict_Car_Brand(image_upload):
    # Convert the PIL image to a NumPy array
    image_array = np.array(image_upload)
    
    # Resize the image to (224, 224)
    image_resized = cv2.resize(image_array, (224, 224))
    
    img_array = img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    
    # Convert to TensorFlow tensor
    image_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    # Predict using the tf.function
    predicted_brand = predict_image(image_tensor)
    label = predicted_brand.numpy()[0].decode()
    return label

# Create and launch the Gradio interface
demo = gr.Interface(
    Predict_Car_Brand, 
    inputs = "image",
    outputs="text",
    title = "Car Brand Predictor",
    description="Upload an image of your Car to predict its brand. (Audi, Hyundai Creta, Mahindra Scorpio, Rolls Royce, Swift, Tata Safari or Toyota Innova )",
    cache_examples=True,
    theme="default",
    allow_flagging="manual",
    flagging_options=["Flag as incorrect", "Flag as inaccurate"],
    analytics_enabled=True,
    batch=False,
    max_batch_size=4,
    allow_duplication=False
)

demo.launch()

