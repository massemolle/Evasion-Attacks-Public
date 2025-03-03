import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

def get_resnet50_model():
    """
    Loads the pre-trained ResNet50 model (trained on ImageNet).
    """
    model = ResNet50(weights='imagenet')
    print("Pre-trained ResNet50 loaded successfully.")
    return model

def load_and_preprocess_image(img_path):
    """
    Loads and preprocesses an image for ResNet50.
    
    The image is resized to 224x224, converted to an array,
    expanded to include the batch dimension, and preprocessed
    using the ResNet50-specific function.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess image (converts RGB to BGR, zero-centers channels, etc.)
    img_array = preprocess_input(img_array)
    return img_array
