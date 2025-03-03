import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import json
from datetime import datetime
import time
from tensorflow.keras.applications.resnet50 import decode_predictions

# Add parent directory to path for imports
sys.path.append('../../')
from model_to_attack.ResNet50_model import get_resnet50_model, load_and_preprocess_image

class AutoPGDAttack:
    def __init__(self, model, epsilon=64/255, step_size=3/255, num_steps=200):
        """
        Initialize Auto-PGD attack with stronger parameters for higher success rate
        
        Args:
            model: The model to attack
            epsilon: Maximum perturbation (significantly increased)
            step_size: Step size for each iteration (increased)
            num_steps: Number of iterations (significantly increased)
        """
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        
        # Create directory for results if it doesn't exist
        self.results_dir = os.path.join(os.path.dirname(__file__), 'attack_results')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def generate_adversarial_example(self, img_array):
        """
        Generate adversarial example using a stronger PGD attack with momentum.
        
        Args:
            img_array: Input image as numpy array (preprocessed for ResNet50)
            
        Returns:
            Tuple: (adversarial image as numpy array, original class, target class, 
                    original confidence, original class name, target class name)
        """
        # Get original prediction
        original_pred = self.model.predict(img_array)
        original_class = np.argmax(original_pred, axis=1)[0]
        original_confidence = float(original_pred[0][original_class])
        original_class_name = decode_predictions(original_pred, top=1)[0][0][1]
        
        # For a targeted attack, select a different class (choose second most likely)
        sorted_indices = np.argsort(original_pred[0])[::-1]
        target_class = sorted_indices[1] if sorted_indices.size > 1 else (original_class + 1) % 1000
        target_class_name = decode_predictions(np.array([np.eye(1000)[target_class]]), top=1)[0][0][1]
        
        # Prepare the input as a tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        # Momentum decay factor
        momentum_decay = 0.9
        
        # Define the attack loop as a tf.function for efficiency
        @tf.function
        def attack_loop(adv_img, grad_momentum):
            for _ in tf.range(self.num_steps):
                with tf.GradientTape() as tape:
                    tape.watch(adv_img)
                    logits = self.model(adv_img)
                    # For targeted attack: maximize the target logit by minimizing its negative
                    target_onehot = tf.one_hot([target_class], depth=1000)
                    loss = -tf.reduce_sum(target_onehot * tf.nn.softmax(logits))
                gradients = tape.gradient(loss, adv_img)
                norm_gradients = gradients / tf.reduce_mean(tf.abs(gradients))
                grad_momentum = momentum_decay * grad_momentum + norm_gradients
                adv_img = adv_img - self.step_size * tf.sign(grad_momentum)
                # Project the perturbation
                perturbation = tf.clip_by_value(adv_img - img_tensor, -self.epsilon, self.epsilon)
                adv_img = tf.clip_by_value(img_tensor + perturbation,
                                           tf.reduce_min(img_tensor),
                                           tf.reduce_max(img_tensor))
            return adv_img, grad_momentum

        best_adv_img = None
        best_adv_loss = float('inf')
        
        # Try multiple random restarts
        for restart in range(3):
            if restart > 0:
                noise = tf.random.uniform(shape=img_array.shape,
                                          minval=-0.1*self.epsilon,
                                          maxval=0.1*self.epsilon,
                                          dtype=tf.float32)
                adv_img_init = tf.clip_by_value(img_tensor + noise,
                                                tf.reduce_min(img_tensor),
                                                tf.reduce_max(img_tensor))
            else:
                adv_img_init = img_tensor
            
            grad_momentum = tf.zeros_like(adv_img_init)
            adv_img_final, _ = attack_loop(adv_img_init, grad_momentum)
            
            # Evaluate the adversarial example for this restart (no .numpy() call needed)
            adv_pred = self.model.predict(adv_img_final)
            current_loss = adv_pred[0][target_class]  # Use the value directly
            if current_loss < best_adv_loss:
                best_adv_loss = current_loss
                best_adv_img = adv_img_final.numpy()
        
        return best_adv_img, original_class, target_class, original_confidence, original_class_name, target_class_name
    
    def _pick_random_images(self, base_dir, max_images, confidence_threshold):
        """
        Randomly pick images from subfolders under 'base_dir' (one folder per class).
        
        Returns a list of tuples: (img_path, img_array, class_idx, confidence)
        """
        class_folders = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
        
        high_confidence_images = []
        max_attempts = 1000
        attempts = 0
        
        while len(high_confidence_images) < max_images and attempts < max_attempts:
            attempts += 1
            
            # Random folder (class)
            random_folder = random.choice(class_folders)
            folder_path = os.path.join(base_dir, random_folder)
            
            # Random image in that folder
            images_in_class = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if not images_in_class:
                continue
            
            random_image = random.choice(images_in_class)
            img_path = os.path.join(folder_path, random_image)
            
            # Load & check confidence
            try:
                img_array = load_and_preprocess_image(img_path)
                pred = self.model.predict(img_array)
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
        Randomly pick 'max_images' from subfolders of 'base_dir' (each subfolder is a class)
        and attack each image if the model's confidence >= confidence_threshold.
        Also computes additional statistics over the predictions.
        """
        # Create timestamp for this attack run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attack_dir = os.path.join(self.results_dir, f"autopgd_attack_{timestamp}")
        os.makedirs(attack_dir, exist_ok=True)
        
        attack_results = {
            "attack_name": "AutoPGD",
            "parameters": {
                "epsilon": float(self.epsilon),
                "step_size": float(self.step_size),
                "num_steps": self.num_steps
            },
            "results": [],
            "total_attack_time_seconds": 0
        }
        
        print("Selecting random high-confidence images...")
        high_confidence_images = self._pick_random_images(
            base_dir=base_dir,
            max_images=max_images,
            confidence_threshold=confidence_threshold
        )
        
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
                
                adv_pred = self.model.predict(adv_img)
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
                
                # Visualization
                fig, axes = plt.subplots(1, 3, figsize=(20, 7))
                def deprocess_image(preprocessed_img):
                    img = preprocessed_img.copy()
                    img = img[0]
                    img = img[:, :, ::-1]  # Convert BGR to RGB
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
                perturbation_vis = (perturbation_vis - np.min(perturbation_vis)) / \
                                   (np.max(perturbation_vis) - np.min(perturbation_vis) + 1e-8)
                axes[2].imshow(perturbation_vis)
                axes[2].set_title(f"Perturbation\nL2 Distance: {l2_dist:.4f}")
                axes[2].axis('off')
                
                plt.suptitle(f"AutoPGD Attack - {'Success' if success else 'Partial Success'}")
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
        
        # Compute additional statistics
        if attack_results["results"]:
            orig_confs = np.array(original_confidences)
            adv_confs = np.array(adversarial_confidences)
            l2_arr = np.array(l2_distances)
            statistics = {
                "original_confidence": {
                    "median": float(np.median(orig_confs)),
                    "mean": float(np.mean(orig_confs)),
                    "std": float(np.std(orig_confs))
                },
                "adversarial_confidence": {
                    "median": float(np.median(adv_confs)),
                    "mean": float(np.mean(adv_confs)),
                    "std": float(np.std(adv_confs))
                },
                "l2_distance": {
                    "median": float(np.median(l2_arr)),
                    "mean": float(np.mean(l2_arr)),
                    "std": float(np.std(l2_arr))
                }
            }
        else:
            statistics = {}
        
        attack_results["summary"]["statistics"] = statistics
        
        # Save JSON
        with open(os.path.join(attack_dir, "attack_results.json"), 'w') as f:
            json.dump(attack_results, f, indent=4)
        
        self._create_results_visualization(high_confidence_images, attack_results, attack_dir)
        
        print("\nAttack Statistics:")
        print(f"Total images processed: {len(high_confidence_images)}")
        print(f"Successful attacks: {successful_attacks}")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Average confidence reduction: {attack_results['summary']['average_confidence_reduction']:.4f}")
        print(f"Average L2 perturbation size: {attack_results['summary']['average_l2_distance']:.4f}")
        print(f"Total attack time: {total_attack_time:.2f} seconds")
        print("Additional Statistics:", json.dumps(statistics, indent=4))
        
        return attack_results
    
    def _create_results_visualization(self, image_data, results, output_dir):
        """Create a detailed visualization of attack results"""
        conf_reductions = [r["confidence_reduction"] for r in results["results"]]
        l2_distances = [r["l2_distance"] for r in results["results"]]
        attack_times = [r["attack_time_seconds"] for r in results["results"]]
        orig_confidences = [r["original_confidence"] for r in results["results"]]
        adv_confidences = [r["confidence_for_original_class"] for r in results["results"]]
        orig_class_names = [r["original_class_name"] for r in results["results"]]
        adv_class_names = [r["adversarial_class_name"] for r in results["results"]]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes[0, 0].hist(conf_reductions, bins=10, color='blue', alpha=0.7)
        axes[0, 0].set_title("Confidence Reduction Distribution")
        axes[0, 0].set_xlabel("Confidence Reduction")
        axes[0, 0].set_ylabel("Number of Images")
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        axes[0, 1].scatter(l2_distances, conf_reductions, alpha=0.7, c='red')
        axes[0, 1].set_title("Confidence Reduction vs. Perturbation Size")
        axes[0, 1].set_xlabel("L2 Distance")
        axes[0, 1].set_ylabel("Confidence Reduction")
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        indices = range(len(orig_confidences))
        width = 0.35
        axes[1, 0].bar(indices, orig_confidences, width, label='Original', color='green')
        axes[1, 0].bar([i + width for i in indices], adv_confidences, width, label='After Attack', color='red')
        axes[1, 0].set_title("Confidence Before and After Attack")
        axes[1, 0].set_xlabel("Image Index")
        axes[1, 0].set_ylabel("Confidence")
        axes[1, 0].set_xticks([i + width/2 for i in indices])
        axes[1, 0].set_xticklabels([str(i) for i in indices])
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        axes[1, 1].scatter(attack_times, l2_distances, alpha=0.7, c='purple')
        axes[1, 1].set_title("Attack Time vs. Perturbation Size")
        axes[1, 1].set_xlabel("Attack Time (seconds)")
        axes[1, 1].set_ylabel("L2 Distance")
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle("AutoPGD Attack Analysis", fontsize=16)
        plt.savefig(os.path.join(output_dir, "attack_analysis.png"))
        plt.close()
        
        plt.figure(figsize=(12, 8))
        for i, (orig, adv) in enumerate(zip(orig_class_names, adv_class_names)):
            plt.plot([0, 1], [i, i], 'k-', alpha=0.3)
            plt.text(-0.1, i, orig, ha='right', va='center')
            plt.text(1.1, i, adv, ha='left', va='center')
            marker = 'ro' if orig != adv else 'bo'
            plt.plot(1, i, marker, markersize=8)
                
        plt.xlim(-0.5, 1.5)
        plt.ylim(-1, len(orig_class_names))
        plt.title("Class Transitions from Original to Adversarial")
        plt.xticks([0, 1], ['Original Class', 'Adversarial Class'])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "class_transitions.png"))
        plt.close()

def main():
    # Load pre-trained ResNet50 model (loaded once)
    model = get_resnet50_model()
    print("Pre-trained ResNet50 loaded successfully.")
    
    # Create the AutoPGD attack
    attack = AutoPGDAttack(
        model=model,
        epsilon=16/255,
        step_size=2/255,
        num_steps=100
    )
    
    # Dataset structure: ImageNet-Mini/images/<class_folder>/<images>
    images_dir = "../../Images/ImageNet-Mini/images"
    
    # Attack random images from subfolders (e.g. 1000 images, confidence threshold 0.5)
    attack_results = attack.attack_images(
        base_dir=images_dir,
        max_images=4,
        confidence_threshold=0.8
    )
    
    # Run the cotation evaluation
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
