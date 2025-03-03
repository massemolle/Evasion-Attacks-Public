import os
# 0 = all logs, 1 = filter out INFO logs, 2 = filter out INFO & WARNING, 3 = filter out all logs except ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
import numpy as np
from model_to_attack import ResNet50_model as ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions

def fgsm_attack(model, image_tensor, label, epsilon=0.01):
    """
    Generates adversarial examples using FGSM on ResNet50.
    
    Parameters:
      model: A pre-trained ResNet50 model.
      image_tensor: The preprocessed image tensor.
      label: One-hot encoded label corresponding to the image.
      epsilon: The perturbation magnitude.
      
    Returns:
      adv_image: The adversarially perturbed image tensor.
    """
    image_tensor = tf.cast(image_tensor, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor)
        loss = tf.keras.losses.categorical_crossentropy(label, predictions)
    
    gradient = tape.gradient(loss, image_tensor)
    signed_grad = tf.sign(gradient)
    # Note: Clipping range should align with the preprocessed input distribution.
    adv_image = image_tensor + epsilon * signed_grad
    # Depending on the preprocessing, valid pixel ranges may vary.
    adv_image = tf.clip_by_value(adv_image, tf.reduce_min(image_tensor), tf.reduce_max(image_tensor))
    return adv_image

def main():
    # Load the pre-trained ResNet50 model.
    model = ResNet50.get_resnet50_model()
    
    # Load and preprocess a sample image.
    # Replace 'sample.jpg' with the path to your image.
    img_path = 'Images/val2017/000000000285.jpg'
    original_image = ResNet50.load_and_preprocess_image(img_path)
    
    # Obtain model predictions for the original image.
    predictions = model.predict(original_image)
    decoded = decode_predictions(predictions, top=1)
    print("Original prediction:", decoded)
    
    # Create a one-hot label based on the top prediction.
    label_index = np.argmax(predictions[0])
    label = tf.one_hot(label_index, 1000)
    label = tf.reshape(label, (1, 1000))
    
    # Apply FGSM attack.
    epsilon = 0.2  # Adjust epsilon to control attack strength.
    adv_image = fgsm_attack(model, original_image, label, epsilon)
    
    # Evaluate model predictions on the adversarial image.
    adv_predictions = model.predict(adv_image)
    decoded_adv = decode_predictions(adv_predictions, top=1)
    print("Adversarial prediction:", decoded_adv)

if __name__ == '__main__':
    main()
