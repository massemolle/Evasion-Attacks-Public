#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf

from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import CarliniL2Method

# Import your ResNet18 model and helper function
from model_to_attack.ResNet18_model import get_resnet18_model, load_and_preprocess_image

def load_images_from_folder(folder, num_images=10):
    """
    Loads up to num_images from the given folder.
    Each image is loaded using load_and_preprocess_image so that it is preprocessed for ResNet18.
    """
    images = []
    filenames = []
    count = 0
    for filename in sorted(os.listdir(folder)):
        if count >= num_images:
            break
        img_path = os.path.join(folder, filename)
        try:
            # Each image is already expanded to shape (1, 224, 224, 3)
            img = load_and_preprocess_image(img_path)
            images.append(img)
            filenames.append(filename)
            count += 1
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    # Stack images along the first dimension (n, 224, 224, 3)
    if images:
        return np.vstack(images), filenames
    else:
        raise ValueError("No images were loaded from the folder.")

def main():
    # Load the pre-trained ResNet18 model.
    # This model currently outputs probabilities.
    model = get_resnet18_model()
    
    # Check if the last layer uses softmax. If so, remove it to obtain logits.
    if hasattr(model.layers[-1], "activation") and model.layers[-1].activation == tf.keras.activations.softmax:
        print("Removing softmax activation to obtain logits for the classifier.")
        model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    else:
        print("Model output is assumed to be logits.")

    # Define clip_values.
    # IMPORTANT: Make sure these match your model's expected input range.
    clip_values = (0.0, 255.0)

    # Use a loss that expects logits.
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Create ART classifier wrapping your TensorFlow model.
    classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        input_shape=(224, 224, 3),
        nb_classes=1000,
        clip_values=clip_values,
    )

    # Load sample images from Images/val2017.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_folder = os.path.join(current_dir, '../../Images/val2017')
    x_test, filenames = load_images_from_folder(images_folder, num_images=5)  # Reduced to 5 images as CW is computationally intensive

    # Display original predictions.
    predictions = classifier.predict(x_test)
    print("Original predictions:")
    original_labels = []
    for i, filename in enumerate(filenames):
        pred_label = np.argmax(predictions[i])
        original_labels.append(pred_label)
        confidence = predictions[i][pred_label]
        print(f"{filename}: Class {pred_label}, Confidence: {confidence:.4f}")

    # Initialize the Carlini & Wagner L2 attack
    attack = CarliniL2Method(
        classifier=classifier,
        confidence=0.0,        # Confidence parameter for adversarial examples
        targeted=False,        # Untargeted attack
        learning_rate=0.01,    # Learning rate for optimization
        binary_search_steps=10, # Number of binary search steps
        max_iter=100,          # Maximum number of iterations
        initial_const=0.01,    # Initial value of the constant c
        max_halving=5,         # Maximum number of halving steps in the line search
        max_doubling=5,        # Maximum number of doubling steps in the line search
        batch_size=1,          # Size of batches
        verbose=True           # Show progress bars
    )

    # Generate adversarial examples
    print("\nGenerating adversarial examples with Carlini & Wagner attack...")
    x_test_adv = attack.generate(x=x_test)

    # Display predictions for adversarial examples
    adv_predictions = classifier.predict(x_test_adv)
    print("\nAdversarial predictions:")
    for i, filename in enumerate(filenames):
        pred_label_adv = np.argmax(adv_predictions[i])
        confidence = adv_predictions[i][pred_label_adv]
        changed = "✓" if pred_label_adv != original_labels[i] else "✗"
        print(f"{filename}: Class {pred_label_adv}, Confidence: {confidence:.4f}, Changed: {changed}")

    # Calculate and display L2 perturbation for each image
    print("\nL2 perturbation statistics:")
    l2_norms = []
    for i, filename in enumerate(filenames):
        l2_norm = np.sqrt(np.sum((x_test_adv[i] - x_test[i])**2))
        l2_norms.append(l2_norm)
        print(f"{filename}: L2 norm = {l2_norm:.4f}")
    
    print(f"Average L2 perturbation: {np.mean(l2_norms):.4f}")

    # Save adversarial images into the attack_results folder
    results_folder = os.path.join(current_dir, 'attack_results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Save both original and adversarial images, along with their difference
    for i, filename in enumerate(filenames):
        # Create a subdirectory for each image to store original, adversarial and difference
        img_folder = os.path.join(results_folder, filename.split('.')[0])
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
            
        # Save original image
        orig_img = np.squeeze(x_test[i])
        orig_img = np.clip(orig_img, clip_values[0], clip_values[1]).astype(np.uint8)
        orig_path = os.path.join(img_folder, f"original_{filename}")
        tf.keras.preprocessing.image.save_img(orig_path, orig_img)
        
        # Save adversarial image
        adv_img = np.squeeze(x_test_adv[i])
        adv_img = np.clip(adv_img, clip_values[0], clip_values[1]).astype(np.uint8)
        adv_path = os.path.join(img_folder, f"cw_adv_{filename}")
        tf.keras.preprocessing.image.save_img(adv_path, adv_img)
        
        # Calculate and save difference image (scaled for visibility)
        diff = np.abs(x_test_adv[i] - x_test[i])
        norm_diff = diff / np.max(diff) * 255 if np.max(diff) > 0 else diff
        diff_path = os.path.join(img_folder, f"diff_{filename}")
        tf.keras.preprocessing.image.save_img(diff_path, norm_diff.astype(np.uint8))
        
        print(f"Saved images to {img_folder}")

    # Create a summary file with attack results
    summary_path = os.path.join(results_folder, "cw_attack_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Carlini & Wagner L2 Attack Summary\n")
        f.write("==================================\n\n")
        
        f.write("Attack Parameters:\n")
        f.write(f"- Confidence: 0.0\n")
        f.write(f"- Targeted: False\n")
        f.write(f"- Max Iterations: 100\n")
        f.write(f"- Learning Rate: 0.01\n\n")
        
        f.write("Attack Results:\n")
        success_count = sum(1 for i in range(len(filenames)) if np.argmax(adv_predictions[i]) != original_labels[i])
        f.write(f"- Success Rate: {success_count}/{len(filenames)} ({success_count/len(filenames)*100:.2f}%)\n")
        f.write(f"- Average L2 Perturbation: {np.mean(l2_norms):.4f}\n\n")
        
        f.write("Individual Results:\n")
        for i, filename in enumerate(filenames):
            orig_label = original_labels[i]
            adv_label = np.argmax(adv_predictions[i])
            orig_conf = predictions[i][orig_label]
            adv_conf = adv_predictions[i][adv_label]
            
            f.write(f"Image: {filename}\n")
            f.write(f"- Original Class: {orig_label} (Confidence: {orig_conf:.4f})\n")
            f.write(f"- Adversarial Class: {adv_label} (Confidence: {adv_conf:.4f})\n")
            f.write(f"- L2 Perturbation: {l2_norms[i]:.4f}\n")
            f.write(f"- Attack Success: {'Yes' if adv_label != orig_label else 'No'}\n\n")
    
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()
