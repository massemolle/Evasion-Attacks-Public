import tensorflow as tf

# classification_models library
try:
    from classification_models.tfkeras import Classifiers
except ImportError:
    raise ImportError("Please install classification-models:\n"
                      "  pip install git+https://github.com/qubvel/classification_models.git\n"
                      "or\n"
                      "  pip install classification-models")

from tensorflow.keras.preprocessing import image
import numpy as np

def get_resnet18_model():
    """
    Loads a pre-trained ResNet18 model (trained on ImageNet) via classification_models.
    """
    ResNet18, preprocess_fn = Classifiers.get('resnet18')
    model = ResNet18(input_shape=(224, 224, 3), weights='imagenet', classes=1000)
    print("Pre-trained ResNet18 loaded successfully.")
    return model

def load_and_preprocess_image(img_path):
    """
    Loads and preprocesses an image for ResNet18.
    The image is resized to 224x224, converted to a NumPy array,
    and then preprocessed with the model-specific preprocess_fn.
    """
    ResNet18, preprocess_fn = Classifiers.get('resnet18')
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # classification_models provides a custom preprocess function
    img_array = preprocess_fn(img_array)
    return img_array
