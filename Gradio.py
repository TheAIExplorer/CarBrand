import os
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import cv2

model_path = "car_classification_model.h5"
best_model = load_model(model_path)

class_labels = ['Audi', 'Hyundai Creta', 'Mahindra Scorpio', 'Rolls Royce', 'Swift', 'Tata Safari', 'Toyota Innova']

@tf.function
def predict_image(image_array):
    prediction = best_model(image_array)
    class_index = tf.argmax(prediction, axis=1)
    predicted_class = tf.gather(class_labels, class_index)
    return predicted_class

def Predict_Car_Brand(image_upload):
    image_array = np.array(image_upload)
    
    image_resized = cv2.resize(image_array, (224, 224))
    
    img_array = img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 

    image_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    predicted_brand = predict_image(image_tensor)
    label = predicted_brand.numpy()[0].decode()
    return label

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

