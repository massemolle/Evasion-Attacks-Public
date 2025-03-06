import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import decode_predictions

# Add parent directory to path for imports
import sys
sys.path.append('../../')
from model_to_attack.ResNet50_model import get_resnet50_model, load_and_preprocess_image

# Import ART packages
from art.attacks.evasion import CarliniL2Method  # For the L2 version
from art.estimators.classification import TensorFlowV2Classifier

class ART_CW_Attack:
    def __init__(self, model, norm='L2', batch_size=1, max_iter=1000, learning_rate=1e-2, confidence=0):
        """
        Initialize the ART-based Carlini-Wagner attack.
        
        Args:
            model: The target model.
            norm: Currently supports 'L2'. (For L_inf consider a different ART attack)
            batch_size: Number of images per batch (typically 1 for individual attacks).
            max_iter: Maximum number of iterations for the attack.
            learning_rate: Step size for the optimization.
            confidence: Confidence parameter (similar to kappa in the original paper).
        """
        self.model = model
        self.norm = norm
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.confidence = confidence
        self.query_count = 0  # ART doesn't count queries in the same way, but you can track if needed.
        
        # Create directory for results if it doesn't exist.
        self.results_dir = os.path.join(os.path.dirname(__file__), 'attack_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Wrap the TensorFlow model with ART's classifier.
        # Note: You might need to adjust clip_values and preprocessing to match your ResNet50 pipeline.
        self.classifier = TensorFlowV2Classifier(
            model=self.model,
            loss_object=tf.keras.losses.CategoricalCrossentropy(),
            input_shape=(224, 224, 3),
            nb_classes=1000,
            clip_values=(0.0, 255.0)
        )
        
        # Instantiate the ART Carlini-Wagner L2 attack
        self.attack = CarliniL2Method(
            estimator=self.classifier,
            confidence=self.confidence,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size
        )

    def model_predict(self, input_data):
        """Wrapper for model prediction (ART classifier)."""
        # ART's classifier.predict does the same
        self.query_count += 1
        return self.classifier.predict(input_data)

    def generate_adversarial_example(self, img_array):
        """
        Generate adversarial example using ART's Carlini-Wagner attack.
        
        Args:
            img_array: Preprocessed input image (expected shape: (1,224,224,3)).
            
        Returns:
            Tuple: (adversarial image as numpy array, original class, target class, 
                    original confidence, original class name, target class name)
        """
        # Get original prediction
        original_pred = self.model_predict(img_array)
        original_class = np.argmax(original_pred, axis=1)[0]
        original_confidence = float(original_pred[0][original_class])
        original_class_name = decode_predictions(original_pred, top=1)[0][0][1]

        # For a targeted attack, we can choose the second most likely class as target.
        sorted_indices = np.argsort(original_pred[0])[::-1]
        target_class = sorted_indices[1] if sorted_indices.size > 1 else (original_class + 1) % 1000
        target_class_name = decode_predictions(np.array([np.eye(1000)[target_class]]), top=1)[0][0][1]
        
        # ART's attack implementation is untargeted by default.
        # For a targeted attack, you need to specify target labels. Here, we set up the target one-hot vector.
        target_onehot = np.zeros((1, 1000))
        target_onehot[0, target_class] = 1.0
        
        # Generate adversarial example. For a targeted attack, pass target_labels parameter.
        adv_img = self.attack.generate(x=img_array, y=target_onehot)
        
        # Evaluate adversarial example
        adv_pred = self.model_predict(adv_img)
        adv_class = np.argmax(adv_pred, axis=1)[0]
        adv_conf = float(adv_pred[0][adv_class])
        adv_conf_for_orig = float(adv_pred[0][original_class])
        
        return adv_img, original_class, target_class, original_confidence, original_class_name, target_class_name

    def _pick_random_images(self, base_dir, max_images, confidence_threshold):
        """
        Randomly pick images from subfolders under 'base_dir' (one folder per class).
        
        Returns:
            List of tuples: (img_path, img_array, class_idx, confidence)
        """
        class_folders = [d for d in os.listdir(base_dir)
                         if os.path.isdir(os.path.join(base_dir, d))]
        
        high_confidence_images = []
        max_attempts = 1000
        attempts = 0
        
        while len(high_confidence_images) < max_images and attempts < max_attempts:
            attempts += 1
            
            random_folder = np.random.choice(class_folders)
            folder_path = os.path.join(base_dir, random_folder)
            
            images_in_class = [f for f in os.listdir(folder_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images_in_class:
                continue
            
            random_image = np.random.choice(images_in_class)
            img_path = os.path.join(folder_path, random_image)
            
            try:
                img_array = load_and_preprocess_image(img_path)
                pred = self.model_predict(img_array)
                class_idx = np.argmax(pred, axis=1)[0]
                confidence = float(pred[0][class_idx])
                if confidence >= confidence_threshold:
                    high_confidence_images.append((img_path, img_array, class_idx, confidence))
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        return high_confidence_images

    def attack_images(self, base_dir, max_images=10, confidence_threshold=0.8):
        """
        Randomly select 'max_images' images from subfolders of 'base_dir'
        and attack each image if its confidence is above the threshold.
        Saves results and visualizations similar to the AutoPGD attack.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attack_dir = os.path.join(self.results_dir, f"ART_CW_attack_{timestamp}")
        os.makedirs(attack_dir, exist_ok=True)
        
        attack_results = {
            "attack_name": f"ART_CW_attack_{self.norm}",
            "parameters": {
                "norm": self.norm,
                "max_iter": self.max_iter,
                "learning_rate": self.learning_rate,
                "confidence": self.confidence
            },
            "results": [],
            "total_attack_time_seconds": 0
        }
        
        print("Selecting random high-confidence images...")
        high_confidence_images = self._pick_random_images(base_dir, max_images, confidence_threshold)
        print(f"Found {len(high_confidence_images)} images with confidence >= {confidence_threshold}")
        
        if not high_confidence_images:
            print("No suitable images found. Exiting.")
            return attack_results
        
        successful_attacks = 0
        l2_distances = []
        confidence_reductions = []
        original_confidences = []
        adversarial_confidences = []
        total_attack_time = 0
        
        for i, (img_path, img_array, orig_class, orig_conf) in enumerate(tqdm(high_confidence_images, desc="Attacking images")):
            try:
                start_time = time.time()
                adv_img, original_class, target_class, original_confidence, original_class_name, target_class_name = \
                    self.generate_adversarial_example(img_array)
                attack_time = time.time() - start_time
                total_attack_time += attack_time
                
                adv_pred = self.model_predict(adv_img)
                adv_class = np.argmax(adv_pred, axis=1)[0]
                adv_class_name = decode_predictions(adv_pred, top=1)[0][0][1]
                adv_conf_for_orig = float(adv_pred[0][original_class])
                adv_conf = float(adv_pred[0][adv_class])
                
                confidence_reduction = original_confidence - adv_conf_for_orig
                confidence_reductions.append(confidence_reduction)
                original_confidences.append(original_confidence)
                adversarial_confidences.append(adv_conf)
                
                l2_dist = np.linalg.norm(img_array.flatten() - adv_img.flatten())
                l2_distances.append(l2_dist)
                
                success = (adv_class != original_class)
                if success:
                    successful_attacks += 1
                
                result = {
                    "image": os.path.basename(img_path),
                    "original_class": int(original_class),
                    "original_class_name": original_class_name,
                    "original_confidence": float(original_confidence),
                    "adversarial_class": int(adv_class),
                    "adversarial_class_name": adv_class_name,
                    "adversarial_confidence": float(adv_conf),
                    "confidence_for_original_class": float(adv_conf_for_orig),
                    "confidence_reduction": float(confidence_reduction),
                    "target_class": int(target_class),
                    "target_class_name": target_class_name,
                    "success": bool(success),
                    "l2_distance": float(l2_dist),
                    "attack_time_seconds": float(attack_time)
                }
                attack_results["results"].append(result)
                
                # Visualization for this image
                fig, axes = plt.subplots(1, 3, figsize=(20, 7))
                def deprocess_image(preprocessed_img):
                    img = preprocessed_img.copy()[0]
                    img = img[:, :, ::-1]  # Convert BGR to RGB if necessary
                    img[:, :, 0] += 103.939
                    img[:, :, 1] += 116.779
                    img[:, :, 2] += 123.68
                    return np.clip(img, 0, 255).astype('uint8')
                
                axes[0].imshow(deprocess_image(img_array))
                axes[0].set_title(f"Original: {original_class_name}\nConfidence: {original_confidence:.4f}")
                axes[0].axis('off')
                
                axes[1].imshow(deprocess_image(adv_img))
                axes[1].set_title(f"Adversarial: {adv_class_name}\nConfidence: {adv_conf:.4f}")
                axes[1].axis('off')
                
                perturbation = adv_img - img_array
                perturbation_vis = perturbation[0]
                perturbation_vis = (perturbation_vis - np.min(perturbation_vis)) / (np.max(perturbation_vis) - np.min(perturbation_vis) + 1e-8)
                axes[2].imshow(perturbation_vis)
                axes[2].set_title(f"Perturbation\nL2 Distance: {l2_dist:.4f}")
                axes[2].axis('off')
                
                plt.suptitle(f"ART CW Attack ({self.norm}) - {'Success' if success else 'Partial Success'}")
                plt.tight_layout()
                plt.savefig(os.path.join(attack_dir, f"attack_result_{i}.png"))
                plt.close()
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        attack_results["total_attack_time_seconds"] = total_attack_time
        success_rate = successful_attacks / len(high_confidence_images)
        attack_results["summary"] = {
            "total_images": len(high_confidence_images),
            "successful_attacks": successful_attacks,
            "success_rate": float(success_rate),
            "average_l2_distance": float(np.mean(l2_distances)) if l2_distances else 0,
            "average_confidence_reduction": float(np.mean(confidence_reductions)) if confidence_reductions else 0,
            "total_attack_time_seconds": float(total_attack_time)
        }
        attack_results["num_model_queries"] = self.query_count
        
        # Save results to JSON
        with open(os.path.join(attack_dir, "attack_results.json"), 'w') as f:
            json.dump(attack_results, f, indent=4)
        
        print("\nAttack Statistics:")
        print(f"Total images processed: {len(high_confidence_images)}")
        print(f"Successful attacks: {successful_attacks}")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Average confidence reduction: {attack_results['summary']['average_confidence_reduction']:.4f}")
        print(f"Average L2 perturbation size: {attack_results['summary']['average_l2_distance']:.4f}")
        print(f"Total attack time: {total_attack_time:.2f} seconds")
        
        return attack_results

def main():
    # Load pre-trained ResNet50 model
    model = get_resnet50_model()
    print("Pre-trained ResNet50 loaded successfully.")
    
    # Create the ART-based CW attack instance (using L2 norm)
    attack = ART_CW_Attack(
        model=model,
        norm='L2',
        batch_size=1,
        max_iter=1000,
        learning_rate=1e-2,
        confidence=0
    )
    
    # Dataset structure: ImageNet-Mini/images/<class_folder>/<images>
    images_dir = "Images/ImageNet-Mini/images"
    
    # Attack random images from subfolders
    attack_results = attack.attack_images(
        base_dir=images_dir,
        max_images=4,
        confidence_threshold=0.8
    )
    
    # Run the cotation evaluation (assuming cotation.py is set up similarly)
    from Evaluation_Metrics.Cotation import CotationTable
    attack_folders = sorted(os.listdir(attack.results_dir))
    if not attack_folders:
        print("No attack results folder found.")
        return
    latest_folder = attack_folders[-1]
    results_json_path = os.path.join(attack.results_dir, latest_folder, "attack_results.json")
    
    if not os.path.exists(results_json_path):
        print("No attack_results.json found; skipping cotation.")
        return
    
    cotation = CotationTable(results_json_path)
    evaluation = cotation.evaluate_attack(
        attacker_knowledge="Proficient", 
        target_knowledge="White-box", 
        equipment="Basic (personal computer)", 
        extrapolation_time="1 week"
    )
    
    print("\nCotation Evaluation:")
    print(json.dumps(evaluation, indent=4))

if __name__ == "__main__":
    main()
