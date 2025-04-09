import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# ------------------------------
#  Utility: Load Original Weights and Task Vector
# ------------------------------
model_name = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load original model and weights
original_model = CLIPModel.from_pretrained(model_name).to(device)
original_weights = {k: v.cpu().clone() for k, v in original_model.state_dict().items()}

# Load normalized task vector
normalized_task_vector = torch.load("task_vector.pt", map_location="cpu")

# ------------------------------
#  Function: Transform State Dictionary
# ------------------------------
def transform_state_dict(alpha, blend):
    """
    Given hyperparameters alpha and blend (0 <= blend <= 1), compute debiased weights as:
        W_debiased = original_weight - ((1 - blend) * alpha) * task_vector
    for each weight in the state_dict.
    """
    new_state_dict = {}
    factor = (1.0 - blend) * alpha
    for k in original_weights:
        # Make sure the task vector has the same shape
        if k in normalized_task_vector:
            new_state_dict[k] = original_weights[k] - factor * normalized_task_vector[k]
        else:
            new_state_dict[k] = original_weights[k].clone()
    return new_state_dict

# ------------------------------
#  Evaluation Dataset and Function
# ------------------------------
class CLIPValDataset(Dataset):
    def __init__(self, json_file, image_folder, processor):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.processor = processor
        self.image_names = list(self.data.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        meta = self.data[image_name]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        # Process the image only; we will handle text separately
        inputs = self.processor(images=image, return_tensors="pt")
        # Squeeze the batch dimension for images
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)
        return inputs, meta["gender"]

def evaluate_model(state_dict, processor, val_json="data/val_data.json", image_folder="img_align_celeba", batch_size=16):
    """
    Loads the provided state_dict into a CLIP model, runs evaluation over the validation set,
    and returns the metrics RA_m, RA_f, RA_avg, and Gender Gap (GG).
    """
    model = CLIPModel.from_pretrained(model_name)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Create validation dataset and dataloader
    val_dataset = CLIPValDataset(val_json, image_folder, processor)
    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Define text prompts for evaluation.
    # Index 0 -> male prompt, Index 1 -> female prompt.
    text_prompts = ['The person in the image is male', 'The person in the image is female']
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
    for key in text_inputs:
        text_inputs[key] = text_inputs[key].to(device)
    
    correct_male = 0
    correct_female = 0
    total_male = 0
    total_female = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images_batch, genders = batch
            # Move image tensors to device
            for key in images_batch:
                images_batch[key] = images_batch[key].to(device)
            # Compute image and text features
            image_features = model.get_image_features(pixel_values=images_batch["pixel_values"])
            text_features = model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"]
            )
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Compute cosine similarity logits
            logits = image_features @ text_features.t()
            preds = torch.argmax(logits, dim=-1)
            for pred, true_gender in zip(preds, genders):
                if true_gender == "male":
                    total_male += 1
                    if pred.item() == 0:  # male prompt has index 0
                        correct_male += 1
                else:
                    total_female += 1
                    if pred.item() == 1:  # female prompt has index 1
                        correct_female += 1
    
    RA_m = correct_male / total_male if total_male > 0 else 0.0
    RA_f = correct_female / total_female if total_female > 0 else 0.0
    RA_avg = (RA_m + RA_f) / 2.0
    GG = abs(RA_m - RA_f)
    return RA_m, RA_f, RA_avg, GG

# ------------------------------
#  Hyperparameter Search over α and Blend
# ------------------------------
def hyperparam_search(processor, num_search=50, lambda_gap=1.0):
    """
    Perform a simple random (or grid) search over α and blend hyperparameters.
    Our loss is defined as:
        loss = -RA_avg + lambda_gap * GG
    so that minimizing loss encourages high average resolution accuracy and a low gender gap.
    
    num_search: number of random trials.
    lambda_gap: weight for the gender gap penalty.
    
    Returns the best hyperparameters and evaluation metrics.
    """
    best_loss = float("inf")
    best_alpha = None
    best_blend = None
    best_metrics = None
    # We search over alpha in [0.1, 1.0] and blend in [0.0, 1.0]
    for i in range(num_search):
        # You can replace random sampling with a grid search if you prefer.
        alpha_candidate = 0.1 + (1.0 - 0.1) * torch.rand(1).item()  # in [0.1, 1.0]
        blend_candidate = torch.rand(1).item()  # in [0.0, 1.0]
        
        # Compute debiased weights using the candidate hyperparameters.
        candidate_state_dict = transform_state_dict(alpha_candidate, blend_candidate)
        # Evaluate on the validation set
        RA_m, RA_f, RA_avg, GG = evaluate_model(candidate_state_dict, processor)
        loss = -RA_avg + lambda_gap * GG
        
        print(f"Trial {i+1:02d}: alpha={alpha_candidate:.4f}, blend={blend_candidate:.4f}, RA_avg={RA_avg:.4f}, GG={GG:.4f}, Loss={loss:.4f}")
        
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha_candidate
            best_blend = blend_candidate
            best_metrics = (RA_m, RA_f, RA_avg, GG)
    
    print("\nBest hyperparameters:")
    print(f"alpha = {best_alpha:.4f}, blend = {best_blend:.4f}")
    print(f"RA_m: {best_metrics[0]:.4f}, RA_f: {best_metrics[1]:.4f}, RA_avg: {best_metrics[2]:.4f}, GG: {best_metrics[3]:.4f}")
    return best_alpha, best_blend, best_metrics

# ------------------------------
#  Main Routine: Learn Hyperparameters and Save Final Model
# ------------------------------
def main():
    os.makedirs("models_final", exist_ok=True)
    
    print("Initializing CLIP processor...")
    processor = CLIPProcessor.from_pretrained(model_name)
    
    print("Starting hyperparameter search...")
    # You can adjust the number of search iterations and λ_gap as needed.
    best_alpha, best_blend, best_metrics = hyperparam_search(processor, num_search=50, lambda_gap=1.0)
    
    # Compute final state_dict using best hyperparameters
    final_state_dict = transform_state_dict(best_alpha, best_blend)
    
    # Save the final state_dict
    final_save_path = os.path.join("models_final", "best.pt")
    torch.save(final_state_dict, final_save_path)
    print(f"Saved final debiased model state dictionary at: {final_save_path}")

if __name__ == "__main__":
    main()
