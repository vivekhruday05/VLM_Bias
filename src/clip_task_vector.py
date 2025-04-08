import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# -------------------------------------------------------
# 1. Load the Dataset and Select Anti-Stereotypical Samples
# -------------------------------------------------------
with open("data/train_data.json", "r") as f:
    dataset = json.load(f)

# Here we assume that anti-stereotypical examples are those that do NOT reinforce the bias.
# Our goal is to improve male resolution accuracy (RA_m) while keeping RA_f high.
# Thus, we select samples with "stereotypical": false.
anti_stereotypical_data = [
    (img, data["overall_caption"]) for img, data in dataset.items() if data["stereotypical"]
]

# -------------------------------------------------------
# 2. Define a Custom Dataset for Anti-Stereotypical Data
# -------------------------------------------------------
class AntiStereotypeDataset(Dataset):
    def __init__(self, data, img_dir):
        """
        data: list of tuples (img_path, caption)
        img_dir: directory where images are stored
        """
        self.data = data
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, caption = self.data[idx]
        return img_path, caption

# -------------------------------------------------------
# 3. Setup the Model, Processor, and Device
# -------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
original_model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Save original weights for later computation of the Task Vector.
original_weights = {k: v.clone() for k, v in original_model.state_dict().items()}

# -------------------------------------------------------
# 4. Create DataLoader with a Custom Collate Function
# -------------------------------------------------------
def collate_fn(batch):
    """
    Given a batch of (img_path, caption) tuples, load the images and process the texts.
    """
    images = []
    texts = []
    for img_path, caption in batch:
        image = Image.open(os.path.join("img_align_celeba", img_path)).convert("RGB")
        images.append(image)
        texts.append(caption)
    return processor(text=texts, images=images, return_tensors="pt", padding=True)

dataset_obj = AntiStereotypeDataset(anti_stereotypical_data, img_dir="img_align_celeba")
dataloader = DataLoader(
    dataset_obj,
    batch_size=32,        # Adjust batch size as needed.
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True
)

# -------------------------------------------------------
# 5. Fine-Tune the Model on Anti-Stereotypical Data Using Cosine Similarity Loss
# -------------------------------------------------------
def train_clip(model, dataloader, epochs=5, lr=5e-6):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # CosineEmbeddingLoss encourages the text and image embeddings to be similar.
    cosine_loss = nn.CosineEmbeddingLoss(margin=0.0)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            # Move the processed batch to device
            batch = {key: value.to(device) for key, value in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize embeddings (as done in the original CLIP training)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Target of 1 means the image and text embeddings should be similar.
            target = torch.ones(image_embeds.size(0)).to(device)
            loss = cosine_loss(image_embeds, text_embeds, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return model

# Fine-tune the model on anti-stereotypical examples.
# (In practice, use more epochs if you have a larger dataset.)
fine_tuned_model = train_clip(original_model, dataloader, epochs=3, lr=5e-6)

# -------------------------------------------------------
# 6. Compute the Task Vector and Apply It to Debias the Model
# -------------------------------------------------------
# Compute the task vector (the change in weights)
fine_tuned_weights = fine_tuned_model.state_dict()
task_vector = {k: fine_tuned_weights[k] - original_weights[k] for k in original_weights.keys()}

normalized_task_vector = {}
for k, v in task_vector.items():
    norm = torch.norm(v)
    normalized_task_vector[k] = v / norm if norm != 0 else v

torch.save(normalized_task_vector, "task_vector.pt")
# To “unlearn” the stereotypical bias, subtract the task vector from the original weights.
# This gives debiased weights that (ideally) retain the ability on non-stereotypical features.
debiased_weights = {k: original_weights[k] - (0.3 * normalized_task_vector[k]) for k in original_weights.keys()}
debiased_weights = {k: ((8.0 * original_weights[k]) + (5.0 * debiased_weights[k]))/13.0 for k in original_weights.keys()}

# Load debiased weights into a new CLIP model instance.
debiased_model = CLIPModel.from_pretrained(model_name)
debiased_model.load_state_dict(debiased_weights)
debiased_model.to(device)

# -------------------------------------------------------
# 7. Save the Debiased Model
# -------------------------------------------------------
torch.save(debiased_model.state_dict(), "models_task_vector/none_frozen.pt")
print("Debiased model saved as debiased_clip_model.pt")
