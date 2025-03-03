import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import decode_predictions
import matplotlib.pyplot as plt
from datetime import datetime

def auto_pgd_attack(model, x, y, epsilon=0.03, steps=40, alpha_max=0.1, alpha_min=0.001, 
                    n_restarts=1, rho=0.75, targeted=False, early_stop=True, verbose=True):
    """
    Enhanced Auto-PGD attack with improved adaptive step size scheduling, momentum, 
    and both targeted and untargeted attack modes.
    
    Parameters:
    -----------
    model : tf.keras.Model
        The target model to attack
    x : tf.Tensor or np.ndarray
        Original input image
    y : tf.Tensor or np.ndarray
        One-hot encoded true label for untargeted attack, or target label for targeted attack
    epsilon : float
        Maximum perturbation size (L-infinity norm)
    steps : int
        Number of optimization steps
    alpha_max : float
        Maximum step size
    alpha_min : float
        Minimum step size
    n_restarts : int
        Number of random restarts
    rho : float
        Momentum factor
    targeted : bool
        If True, perform targeted attack (minimize loss), otherwise untargeted (maximize loss)
    early_stop : bool
        If True, stop when prediction changes
    verbose : bool
        If True, print progress information
        
    Returns:
    --------
    np.ndarray or tf.Tensor
        Adversarial example (same type as input x)
    dict
        Attack statistics (success, iterations, confidence change)
    """
    # Convert inputs to TensorFlow tensors if they're not already
    if not isinstance(x, tf.Tensor):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
    if not isinstance(y, tf.Tensor):
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        
    min_val = tf.reduce_min(x)
    max_val = tf.reduce_max(x)
    
    best_adv = x
    best_loss = -float('inf') if not targeted else float('inf')
    attack_stats = {'iterations': 0, 'loss_progress': []}
    
    # Loss function: negative for targeted attack (minimize), positive for untargeted (maximize)
    loss_multiplier = -1.0 if targeted else 1.0
    
    for restart in range(n_restarts):
        if verbose and n_restarts > 1:
            print(f"Restart {restart+1}/{n_restarts}")
            
        # Initialize with random perturbation within epsilon
        perturbation = tf.random.uniform(tf.shape(x), minval=-epsilon, maxval=epsilon)
        x_adv = tf.clip_by_value(x + perturbation, min_val, max_val)
        
        # Cosine annealing schedule for step size
        iter_steps = np.arange(1, steps + 1)
        alpha_schedule = alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + np.cos(np.pi * iter_steps / steps))
        
        prev_grad = tf.zeros_like(x)
        prev_loss = None
        
        for step in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                predictions = model(x_adv)
                
                # For targeted attack, we want to maximize probability of target class
                # For untargeted, we want to minimize probability of true class
                loss = tf.keras.losses.categorical_crossentropy(y, predictions)
                
            # Get gradient and normalize
            grad = tape.gradient(loss, x_adv)
            
            # Add momentum
            grad = rho * prev_grad + (1 - rho) * grad
            prev_grad = grad
            
            # Use sign of gradient
            signed_grad = tf.sign(grad)
            
            # Update with adaptive step size from schedule
            alpha = alpha_schedule[step]
            x_adv = x_adv + loss_multiplier * alpha * signed_grad
            
            # Project onto epsilon-ball and valid pixel range
            x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)
            x_adv = tf.clip_by_value(x_adv, min_val, max_val)
            
            # Evaluate current progress
            current_preds = model(x_adv)
            current_loss = tf.reduce_mean(loss).numpy()
            attack_stats['loss_progress'].append(current_loss)
            
            if verbose and (step % 10 == 0 or step == steps - 1):
                print(f"Step {step+1}/{steps}, Loss: {current_loss:.4f}, Step size: {alpha:.4f}")
            
            # Early stopping if prediction changed
            if early_stop:
                orig_class = tf.argmax(model(x), axis=1)
                curr_class = tf.argmax(current_preds, axis=1)
                if targeted and curr_class == tf.argmax(y, axis=1):
                    print("Target achieved. Early stopping.")
                    break
                elif not targeted and curr_class != orig_class:
                    print("Classification changed. Early stopping.")
                    break
        
        # Final evaluation
        final_preds = model(x_adv)
        final_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, final_preds))
        
        if (not targeted and final_loss > best_loss) or (targeted and final_loss < best_loss):
            best_loss = final_loss
            best_adv = x_adv
            attack_stats['iterations'] = step + 1
    
    # Compute success: different class for untargeted, target class for targeted
    orig_class = tf.argmax(model(x), axis=1)
    adv_class = tf.argmax(model(best_adv), axis=1)
    
    if targeted:
        target_class = tf.argmax(y, axis=1)
        attack_stats['success'] = bool((adv_class == target_class).numpy()[0])
    else:
        attack_stats['success'] = bool((adv_class != orig_class).numpy()[0])
    
    # Confidence changes
    attack_stats['orig_confidence'] = tf.reduce_max(tf.nn.softmax(model(x))).numpy()
    attack_stats['adv_confidence'] = tf.reduce_max(tf.nn.softmax(model(best_adv))).numpy()
    
    # Convert result back to numpy if input was numpy
    if not isinstance(x, tf.Tensor):
        best_adv = best_adv.numpy()
    
    return best_adv, attack_stats

def visualize_attack(original_img, adv_img, orig_label, adv_label, orig_conf, adv_conf, epsilon, save_path=None):
    """
    Visualizes the original and adversarial images with their predictions and the perturbation.
    
    Parameters:
    -----------
    original_img : np.ndarray
        Original input image
    adv_img : np.ndarray
        Adversarial example
    orig_label : str
        Original classification label
    adv_label : str
        Adversarial classification label
    orig_conf : float
        Original classification confidence
    adv_conf : float
        Adversarial classification confidence
    epsilon : float
        Maximum perturbation size used
    save_path : str, optional
        Path to save the visualization
    """
    # Convert to numpy if needed
    if isinstance(original_img, tf.Tensor):
        original_img = original_img.numpy()
    if isinstance(adv_img, tf.Tensor):
        adv_img = adv_img.numpy()
    
    # Convert from batched to single image if needed
    if len(original_img.shape) == 4:
        original_img = original_img[0]
    if len(adv_img.shape) == 4:
        adv_img = adv_img[0]
    
    # Calculate perturbation and its magnitude
    perturbation = adv_img - original_img
    perturbation_mag = np.abs(perturbation)
    l_inf_norm = np.max(perturbation_mag)
    
    # For visualization: normalize perturbation to [0,1] range
    # Add 0.5 to center at gray (no change)
    vis_perturbation = perturbation / (2 * epsilon) + 0.5
    
    # Create heatmap of perturbation magnitude
    perturbation_heatmap = np.mean(perturbation_mag, axis=2)  # Average across channels
    perturbation_heatmap = perturbation_heatmap / np.max(perturbation_heatmap)  # Normalize to [0,1]
    
    # Convert to displayable range if needed
    if np.max(original_img) <= 1.0:
        orig_display = np.clip(original_img, 0, 1)
        adv_display = np.clip(adv_img, 0, 1)
    else:
        orig_display = np.clip(original_img / 255.0, 0, 1)
        adv_display = np.clip(adv_img / 255.0, 0, 1)
    
    # Set up the figure
    plt.figure(figsize=(16, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(orig_display)
    plt.title(f"Original: {orig_label}\nConfidence: {orig_conf:.2%}")
    plt.axis('off')
    
    # Adversarial image
    plt.subplot(2, 3, 2)
    plt.imshow(adv_display)
    plt.title(f"Adversarial: {adv_label}\nConfidence: {adv_conf:.2%}")
    plt.axis('off')
    
    # Difference image
    plt.subplot(2, 3, 3)
    plt.imshow(vis_perturbation)
    plt.title(f"Perturbation\nL∞ norm: {l_inf_norm:.4f}")
    plt.axis('off')
    
    # Perturbation heatmap
    plt.subplot(2, 3, 5)
    plt.imshow(perturbation_heatmap, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Perturbation Magnitude")
    plt.axis('off')
    
    # Add attack parameters
    plt.subplot(2, 3, 6)
    plt.axis('off')
    info_text = (
        f"Attack Parameters:\n"
        f"ε (epsilon): {epsilon}\n"
        f"Attack success: {'Yes' if orig_label != adv_label else 'No'}\n"
        f"Confidence change: {orig_conf:.2%} → {adv_conf:.2%}\n"
        f"Confidence drop: {orig_conf - adv_conf:.2%}"
    )
    plt.text(0.1, 0.5, info_text, fontsize=12, va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def save_attack_results(original_img, adv_img, img_path, attack_stats, epsilon, output_dir='results'):
    """
    Saves the attack results, including images and statistics.
    
    Parameters:
    -----------
    original_img : np.ndarray or tf.Tensor
        Original input image
    adv_img : np.ndarray or tf.Tensor
        Adversarial example
    img_path : str
        Path to the original image
    attack_stats : dict
        Statistics from the attack
    epsilon : float
        Maximum perturbation size used
    output_dir : str
        Directory to save results
    """
    # Convert tensors to numpy if needed
    if isinstance(original_img, tf.Tensor):
        original_img = original_img.numpy()
    if isinstance(adv_img, tf.Tensor):
        adv_img = adv_img.numpy()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = os.path.basename(img_path).split('.')[0]
    result_dir = os.path.join(output_dir, f"{img_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Save original and adversarial images
    np.save(os.path.join(result_dir, "original.npy"), original_img)
    np.save(os.path.join(result_dir, "adversarial.npy"), adv_img)
    
    # Save attack statistics
    with open(os.path.join(result_dir, "attack_stats.txt"), 'w') as f:
        f.write(f"Original image: {img_path}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Attack success: {attack_stats['success']}\n")
        f.write(f"Iterations needed: {attack_stats['iterations']}\n")
        f.write(f"Original confidence: {attack_stats['orig_confidence']:.4f}\n")
        f.write(f"Adversarial confidence: {attack_stats['adv_confidence']:.4f}\n")
        f.write(f"Confidence change: {attack_stats['orig_confidence'] - attack_stats['adv_confidence']:.4f}\n")
    
    # Save loss progress
    np.save(os.path.join(result_dir, "loss_progress.npy"), np.array(attack_stats['loss_progress']))
    
    return result_dir

def evaluate_attack_difficulty(attack_stats, epsilon):
    """
    Evaluates the difficulty of the attack based on various metrics.
    
    Parameters:
    -----------
    attack_stats : dict
        Statistics from the attack
    epsilon : float
        Maximum perturbation size used
        
    Returns:
    --------
    dict
        Difficulty metrics
    """
    # Difficulty metrics
    difficulty_metrics = {
        'success': attack_stats['success'],
        'iterations_required': attack_stats['iterations'],
        'confidence_drop': attack_stats['orig_confidence'] - attack_stats['adv_confidence'],
        'epsilon': epsilon,
    }
    
    # Normalized difficulty score (0-10 scale)
    # Higher score means more difficult to attack (more secure)
    if attack_stats['success']:
        # If attack succeeded, base difficulty on iterations and epsilon
        iter_factor = min(1.0, attack_stats['iterations'] / 100)  # Normalize iterations
        epsilon_factor = min(1.0, epsilon / 0.3)  # Normalize epsilon
        confidence_factor = min(1.0, attack_stats['adv_confidence'] / attack_stats['orig_confidence'])
        
        # Combined score: higher is more difficult to attack
        difficulty_score = (1 - iter_factor) * 3 + (1 - epsilon_factor) * 5 + confidence_factor * 2
    else:
        # If attack failed, model is more robust
        difficulty_score = 8 + min(2, epsilon * 10)  # Max score of 10
    
    difficulty_metrics['difficulty_score'] = difficulty_score
    
    # Interpretable difficulty level
    if difficulty_score < 2:
        difficulty_metrics['difficulty_level'] = "Very Low (Trivial to attack)"
    elif difficulty_score < 4:
        difficulty_metrics['difficulty_level'] = "Low (Easy to attack)"
    elif difficulty_score < 6:
        difficulty_metrics['difficulty_level'] = "Moderate (Requires optimization)"
    elif difficulty_score < 8:
        difficulty_metrics['difficulty_level'] = "High (Requires significant effort)"
    else:
        difficulty_metrics['difficulty_level'] = "Very High (Highly resistant to attacks)"
    
    return difficulty_metrics

def main():
    from model_to_attack.ResNet50_model import get_resnet50_model, load_and_preprocess_image
    
    # Attack parameters (configurable)
    epsilon = 0.05  # Maximum perturbation size
    steps = 100  # Maximum iterations
    alpha_max = 0.01  # Maximum step size
    alpha_min = 0.001  # Minimum step size
    n_restarts = 2  # Number of random restarts
    rho = 0.75  # Momentum factor
    confidence_threshold = 0.7  # Only process images with original prediction confidence >= 70%
    target_count = 5  # Number of images to process
    targeted = False  # Set to True for targeted attacks
    
    # Output directory
    output_dir = 'attack_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = get_resnet50_model()
    print("Model loaded successfully.\n")
    
    # Get image files
    dataset_dir = 'Images/val2017'
    image_files = [os.path.join(dataset_dir, f)
                   for f in os.listdir(dataset_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    random.shuffle(image_files)
    stats = []
    count = 0
    
    # Summary statistics storage
    all_attack_stats = []
    all_difficulty_metrics = []
    
    for img_path in image_files:
        if count >= target_count:
            break
        
        print(f"\n{'='*50}")
        print(f"Processing image: {img_path}")
        try:
            x_orig = load_and_preprocess_image(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue
        
        # Get original prediction
        orig_preds = model.predict(x_orig)
        decoded_orig = decode_predictions(orig_preds, top=1)[0]
        orig_confidence = decoded_orig[0][2]
        print(f"Original prediction: {decoded_orig[0][1]} ({orig_confidence:.2%})")
        
        if orig_confidence < confidence_threshold:
            print(f"Skipping image due to low confidence (<{confidence_threshold*100:.0f}%).")
            continue
        
        # Get true label
        label_index = np.argmax(orig_preds[0])
        y_true = tf.one_hot(label_index, 1000)
        y_true = tf.reshape(y_true, (1, 1000))
        
        # For targeted attacks, select a different target class
        if targeted:
            # Pick a random target class different from the original
            non_target_indices = [i for i in range(1000) if i != label_index]
            target_index = random.choice(non_target_indices)
            y_target = tf.one_hot(target_index, 1000)
            y_target = tf.reshape(y_target, (1, 1000))
            label_to_use = y_target
            print(f"Target class index: {target_index}")
        else:
            label_to_use = y_true
        
        # Run attack
        print(f"Running Auto-PGD attack with epsilon={epsilon}, steps={steps}, restarts={n_restarts}")
        adv_x, attack_stats = auto_pgd_attack(
            model, x_orig, label_to_use,
            epsilon=epsilon,
            steps=steps,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            n_restarts=n_restarts,
            rho=rho,
            targeted=targeted,
            verbose=True
        )
        
        # Get adversarial prediction
        adv_preds = model.predict(adv_x)
        decoded_adv = decode_predictions(adv_preds, top=1)[0]
        adv_confidence = decoded_adv[0][2]
        print(f"Adversarial prediction: {decoded_adv[0][1]} ({adv_confidence:.2%})")
        
        # Save results
        result_dir = save_attack_results(
            x_orig, adv_x, img_path, 
            attack_stats, epsilon, 
            output_dir=output_dir
        )
        
        # Evaluate attack difficulty
        difficulty_metrics = evaluate_attack_difficulty(attack_stats, epsilon)
        print(f"Attack difficulty score: {difficulty_metrics['difficulty_score']:.2f}/10")
        print(f"Difficulty level: {difficulty_metrics['difficulty_level']}")
        
        # Visualize and save
        vis_save_path = os.path.join(result_dir, "visualization.png")
        visualize_attack(
            x_orig, adv_x,
            decoded_orig[0][1], decoded_adv[0][1],
            orig_confidence, adv_confidence,
            epsilon,
            save_path=vis_save_path
        )
        
        # Store stats
        stats.append({
            'img_path': img_path,
            'orig_class': decoded_orig[0][1],
            'orig_confidence': orig_confidence,
            'adv_class': decoded_adv[0][1],
            'adv_confidence': adv_confidence,
            'changed': (decoded_orig[0][0] != decoded_adv[0][0]),
            'iterations': attack_stats['iterations'],
            'difficulty_score': difficulty_metrics['difficulty_score']
        })
        
        all_attack_stats.append(attack_stats)
        all_difficulty_metrics.append(difficulty_metrics)
        count += 1
    
    # Aggregated statistics
    if stats:
        num_attacked = len(stats)
        num_changed = sum(1 for s in stats if s['changed'])
        success_rate = num_changed / num_attacked
        avg_orig_conf = np.mean([s['orig_confidence'] for s in stats])
        avg_adv_conf = np.mean([s['adv_confidence'] for s in stats])
        avg_iters = np.mean([s['iterations'] for s in stats])
        avg_difficulty = np.mean([s['difficulty_score'] for s in stats])
        
        print("\n" + "="*50)
        print("--- Summary Statistics ---")
        print(f"Number of images attacked: {num_attacked}")
        print(f"Number of images with changed prediction: {num_changed}")
        print(f"Attack success rate: {success_rate:.2%}")
        print(f"Average original confidence: {avg_orig_conf:.2%}")
        print(f"Average adversarial confidence: {avg_adv_conf:.2%}")
        print(f"Average iterations required: {avg_iters:.1f}/{steps}")
        print(f"Average attack difficulty score: {avg_difficulty:.2f}/10")
        
        # Create summary visualization
        plt.figure(figsize=(10, 8))
        
        # Success rate
        plt.subplot(2, 2, 1)
        plt.pie([num_changed, num_attacked - num_changed], 
                labels=['Success', 'Failure'],
                autopct='%1.1f%%',
                colors=['#ff9999', '#66b3ff'])
        plt.title('Attack Success Rate')
        
        # Confidence change
        plt.subplot(2, 2, 2)
        plt.bar(['Original', 'Adversarial'], [avg_orig_conf, avg_adv_conf])
        plt.ylim(0, 1)
        plt.title('Average Confidence')
        plt.ylabel('Confidence')
        
        # Difficulty distribution
        plt.subplot(2, 2, 3)
        difficulty_scores = [s['difficulty_score'] for s in stats]
        plt.hist(difficulty_scores, bins=5, range=(0, 10), alpha=0.7)
        plt.xlabel('Difficulty Score')
        plt.ylabel('Count')
        plt.title('Attack Difficulty Distribution')
        
        # Iterations required
        plt.subplot(2, 2, 4)
        iterations = [s['iterations'] for s in stats]
        plt.hist(iterations, bins=10, alpha=0.7)
        plt.xlabel('Iterations Required')
        plt.ylabel('Count')
        plt.title('Attack Efficiency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'summary_statistics.png'))
        plt.show()
        
        # Save overall summary
        with open(os.path.join(output_dir, 'attack_summary.txt'), 'w') as f:
            f.write("=== Auto-PGD Attack Summary ===\n\n")
            f.write(f"Attack parameters:\n")
            f.write(f"- Epsilon: {epsilon}\n")
            f.write(f"- Steps: {steps}\n")
            f.write(f"- Restarts: {n_restarts}\n")
            f.write(f"- Targeted: {targeted}\n\n")
            
            f.write(f"Results:\n")
            f.write(f"- Images attacked: {num_attacked}\n")
            f.write(f"- Success rate: {success_rate:.2%}\n")
            f.write(f"- Average confidence drop: {avg_orig_conf - avg_adv_conf:.2%}\n")
            f.write(f"- Average iterations: {avg_iters:.1f}\n")
            f.write(f"- Average difficulty score: {avg_difficulty:.2f}/10\n\n")
            
            f.write("Individual results:\n")
            for i, s in enumerate(stats):
                f.write(f"Image {i+1}: {s['img_path']}\n")
                f.write(f"  Original: {s['orig_class']} ({s['orig_confidence']:.2%})\n")
                f.write(f"  Adversarial: {s['adv_class']} ({s['adv_confidence']:.2%})\n")
                f.write(f"  Success: {s['changed']}\n")
                f.write(f"  Difficulty score: {s['difficulty_score']:.2f}/10\n\n")
    else:
        print("No images met the confidence threshold for attack.")

if __name__ == '__main__':
    main()