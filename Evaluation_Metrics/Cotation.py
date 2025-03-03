import os
import json
from datetime import datetime
import psutil

try:
    import GPUtil
except ImportError:
    GPUtil = None

class CotationTable:
    """
    CotationTable evaluates the difficulty of an evasion attack based on several criteria.
    
    Evaluation Criteria:
    
    1. Elapsed Time / Feasibility:
       - Considers the time needed to mount the attack, the attack’s success rate, 
         the average L2 perturbation size, and other relevant metrics.
       - Feasibility is classified as High, Medium, or Low based on these factors.
       
    2. Attacker Knowledge:
       - Laymen: Can follow publicly available instructions and operate off-the-shelf tools.
       - Proficient: Has basic training/experience with standard attack techniques.
       - Expert: Possesses deep technical skills and can implement complex or novel attacks.
       - Multiple experts: A coordinated team or a single individual with interdisciplinary expertise.
       
    3. Target Knowledge:
       - White-box: Complete access to the target (e.g., internal model details) which might be harder to obtain.
       - Gray-box: Partial knowledge about the target.
       - Black-box: Limited knowledge; only observable input/output behavior.
       
    4. Equipment:
       - Evaluates the hardware used. In this example, we assume a basic computer, but more powerful 
         equipment (high-end PC, server/cloud resources, specialized GPU clusters) would affect the attack's feasibility.
       
    5. Extrapolation Time:
       - An estimation of the time needed to build and perform the attack on a more complex system.
       - Options include: "1 day", "1 week", "Multiple weeks", "Months".
       
    Additional Metrics Included:
       - GPU/CPU consumption (current snapshot).
       - Number of model queries performed during the attack (if reported).
       - Attack parameters (epsilon, step_size, num_steps)
       
    The module calculates a total difficulty score normalized to a 0-10 scale (with 10 being the most difficult)
    and produces a detailed rating that includes both the numeric score and qualitative considerations.
    """
    
    # Define rating scales for each evaluation criterion
    RATING_SCALE = {
        #time for the attack to be a success
        "elapsed_time": {
            "Very Fast (< 1 min)": 1,
            "Fast (1-10 min)": 2,
            "Moderate (10-30 min)": 3,
            "Slow (30-60 min)": 4,
            "Very Slow (> 60 min)": 5
        },
        "attacker_knowledge": {
            "Laymen": 1,
            "Proficient": 2,
            "Expert": 3,
            "Multiple experts": 4
        },
        "target_knowledge": {
            "White-box": 1,
            "Gray-box": 2,
            "Black-box": 3
        },
        "equipment": {
            "Basic (personal computer)": 1,
            "Moderate (high-end PC/workstation)": 2,
            "Advanced (server/cloud resources)": 3,
            "Specialized (GPU cluster/supercomputer)": 4
        },
        "extrapolation_time": {
            "1 day": 1,
            "1 week": 2,
            "Multiple weeks": 3,
            "Months": 4
        }
    }
    
    def __init__(self, attack_results_path):
        """
        Initialize the CotationTable by loading attack results from a JSON file.
        
        Args:
            attack_results_path (str): Path to the JSON file containing attack results.
        """
        self.results = None
        self.attack_results_path = attack_results_path
        self.load_attack_results()
        self.resource_usage = self._get_resource_usage()
    
    def load_attack_results(self):
        """
        Loads the attack results from the specified JSON file.
        """
        if not os.path.exists(self.attack_results_path):
            raise FileNotFoundError(f"Attack results file not found: {self.attack_results_path}")
        
        with open(self.attack_results_path, 'r') as f:
            self.results = json.load(f)
        
        print(f"Loaded attack results for {self.results.get('attack_name', 'Unknown Attack')}")
    
    def _get_resource_usage(self):
        """
        Get a snapshot of current resource usage: CPU, memory and (if available) GPU.
        
        Returns:
            dict: Resource usage metrics.
        """
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        mem_usage = memory.percent
        
        gpu_info = {}
        if GPUtil is not None:
            gpus = GPUtil.getGPUs()
            if gpus:
                # For simplicity, we record the first GPU's usage
                gpu = gpus[0]
                gpu_info = {
                    "gpu_id": gpu.id,
                    "gpu_load": gpu.load * 100,
                    "gpu_memory_total": gpu.memoryTotal,
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_util": gpu.memoryUtil * 100
                }
        return {
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": mem_usage,
            "gpu_info": gpu_info
        }
    
    def evaluate_feasibility(self):
        """
        Evaluate the attack's feasibility based on elapsed time, success rate, and L2 perturbation.
        
        Returns:
            dict: A dictionary containing the feasibility ratings and corresponding scores.
        """
        summary = self.results.get("summary", {})
        total_time = summary.get("total_attack_time_seconds", 0)
        success_rate = summary.get("success_rate", 0)
        avg_l2 = summary.get("average_l2_distance", 0)
        avg_conf_reduction = summary.get("average_confidence_reduction", 0)
        
        # Determine elapsed time rating
        if total_time < 60:
            time_rating = "Very Fast (< 1 min)"
        elif total_time < 600:
            time_rating = "Fast (1-10 min)"
        elif total_time < 1800:
            time_rating = "Moderate (10-30 min)"
        elif total_time < 3600:
            time_rating = "Slow (30-60 min)"
        else:
            time_rating = "Very Slow (> 60 min)"
        
        # Evaluate feasibility based on success rate and perturbation size
        if success_rate > 0.8:
            success_rating = "Excellent"
        elif success_rate > 0.5:
            success_rating = "Good"
        elif success_rate > 0.2 or avg_conf_reduction > 0.5:
            success_rating = "Moderate"
        elif success_rate > 0 or avg_conf_reduction > 0.2:
            success_rating = "Limited"
        else:
            success_rating = "Poor"
        
        if avg_l2 < 10:
            perturbation_rating = "Imperceptible"
        elif avg_l2 < 30:
            perturbation_rating = "Slightly visible"
        elif avg_l2 < 50:
            perturbation_rating = "Noticeable"
        elif avg_l2 < 100:
            perturbation_rating = "Obvious"
        else:
            perturbation_rating = "Very visible"
        
        overall_feasibility = (
            "High" if (success_rating in ["Excellent", "Good"] and perturbation_rating in ["Imperceptible", "Slightly visible"]) 
            else "Medium" if (success_rating == "Moderate" or perturbation_rating == "Noticeable") 
            else "Low"
        )
        
        return {
            "time_rating": time_rating,
            "time_score": self.RATING_SCALE["elapsed_time"].get(time_rating, 0),
            "success_rating": success_rating,
            "perturbation_rating": perturbation_rating,
            "overall_feasibility": overall_feasibility
        }
    
    def evaluate_attack(self, attacker_knowledge="Proficient", target_knowledge="White-box",
                        equipment="Basic (personal computer)", extrapolation_time="1 week"):
        """
        Evaluate the attack using multiple criteria and produce an overall difficulty rating.
        
        Args:
            attacker_knowledge (str): Attacker's knowledge level.
            target_knowledge (str): Level of target knowledge.
            equipment (str): Equipment used to perform the attack.
            extrapolation_time (str): Estimated time to adapt the attack for a more complex system.
        
        Returns:
            dict: A comprehensive evaluation including difficulty factors, feasibility metrics,
                  overall difficulty level and score, resource usage, and attack parameters.
        """
        feasibility = self.evaluate_feasibility()
        
        # Retrieve scores from the defined rating scales
        attacker_score = self.RATING_SCALE["attacker_knowledge"].get(attacker_knowledge, 0)
        target_score = self.RATING_SCALE["target_knowledge"].get(target_knowledge, 0)
        equipment_score = self.RATING_SCALE["equipment"].get(equipment, 0)
        extrapolation_score = self.RATING_SCALE["extrapolation_time"].get(extrapolation_time, 0)
        
        total_score = feasibility["time_score"] + attacker_score + target_score + equipment_score + extrapolation_score
        max_possible = 5 + 4 + 3 + 4 + 4
        normalized_score = (total_score / max_possible) * 10
        
        if normalized_score < 3:
            difficulty_level = "Very Easy"
        elif normalized_score < 5:
            difficulty_level = "Easy"
        elif normalized_score < 7:
            difficulty_level = "Moderate"
        elif normalized_score < 9:
            difficulty_level = "Difficult"
        else:
            difficulty_level = "Very Difficult"
        
        evaluation = {
            "attack_name": self.results.get("attack_name", "Unknown Attack"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attack_parameters": self.results.get("parameters", {}),
            "resource_usage": self.resource_usage,
            "model_queries": self.results.get("num_model_queries", "Not Reported"),
            "feasibility": feasibility,
            "difficulty_factors": {
                "attacker_knowledge": {
                    "level": attacker_knowledge,
                    "score": attacker_score,
                    "description": self._attacker_knowledge_description(attacker_knowledge)
                },
                "target_knowledge": {
                    "level": target_knowledge,
                    "score": target_score,
                    "description": self._target_knowledge_description(target_knowledge)
                },
                "equipment": {
                    "level": equipment,
                    "score": equipment_score
                },
                "extrapolation_time": {
                    "level": extrapolation_time,
                    "score": extrapolation_score
                }
            },
            "overall_difficulty": {
                "score": normalized_score,
                "level": difficulty_level,
                "considerations": self._generate_considerations(
                    feasibility["overall_feasibility"],
                    difficulty_level,
                    attacker_knowledge,
                    target_knowledge
                )
            }
        }
        
        return evaluation
    
    def _attacker_knowledge_description(self, level):
        descriptions = {
            "Laymen": "An attacker able to follow publicly available instructions and rely on off-the-shelf tools.",
            "Proficient": "An attacker with basic knowledge, training, or experience with standard attack techniques.",
            "Expert": "An attacker with deep technical skills, capable of executing complex attacks and developing custom solutions.",
            "Multiple experts": "A team or individual with interdisciplinary expertise across multiple domains."
        }
        return descriptions.get(level, "Unknown attacker knowledge level")
    
    def _target_knowledge_description(self, level):
        descriptions = {
            "White-box": "Complete knowledge of the target system, though access to sensitive information (e.g., model internals) may be restricted.",
            "Gray-box": "Partial knowledge of the target system, with some internal details remaining unknown.",
            "Black-box": "Limited knowledge; only the input-output behavior is available without insight into internal workings."
        }
        return descriptions.get(level, "Unknown target knowledge level")
    
    def _generate_considerations(self, feasibility, difficulty, attacker_knowledge, target_knowledge):
        considerations = []
        if feasibility == "High":
            considerations.append("The attack is highly feasible based on its execution time and minimal perturbations.")
        elif feasibility == "Medium":
            considerations.append("The attack has moderate feasibility with balanced performance metrics.")
        else:
            considerations.append("The attack exhibits low feasibility due to extended execution time or significant perturbations.")
        
        if difficulty in ["Very Easy", "Easy"]:
            considerations.append("Overall difficulty is low, requiring minimal expertise and basic equipment.")
        elif difficulty == "Moderate":
            considerations.append("Overall difficulty is moderate; minor improvements or increased resources might ease the attack.")
        else:
            considerations.append("Overall difficulty is high, demanding significant expertise, advanced equipment, and careful planning.")
        
        considerations.append(f"Attacker knowledge: {attacker_knowledge}. Target evaluation: {target_knowledge}.")
        return " ".join(considerations)

if __name__ == "__main__":
    # Example usage:
    attack_results_file = os.path.join(
        os.path.dirname(__file__),
        "..", "Attacks", "Auto_PGD_Attack", "attack_results", "attack_results.json"
    )
    
    cotation = CotationTable(attack_results_file)
    evaluation = cotation.evaluate_attack(
        attacker_knowledge="Proficient",
        target_knowledge="White-box",
        equipment="Basic (personal computer)",
        extrapolation_time="1 week"
    )
    
    print(json.dumps(evaluation, indent=4))
