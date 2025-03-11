import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

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
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)
        return inputs, meta["gender"]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_json = "../data/val_data.json"
    image_folder = "../data/img_align_celeba"

    # # Load the fine-tuned model and processor
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load the fine-tuned model and processor
    model = CLIPModel.from_pretrained("fine_tuned_clip_hf")
    processor = CLIPProcessor.from_pretrained("fine_tuned_clip_hf")

    model.to(device)
    model.eval()

    # Create the validation dataset and DataLoader
    val_dataset = CLIPValDataset(val_json, image_folder, processor)
    dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Define text prompts for gender classification
    # text_prompts = ["The gender of the person in the image is Male", "The gender of the person in the image is Female"]
    # text_prompts = ["The pronoun for the person in the image is he", "The pronoun for the person in the image is he"]
    text_prompts_1 = ["male", "female"]
    text_prompts_2 = ["him", "her"]
    text_prompts_3 = ["he", "she"]
    text_prompts_4 = ['The person in the image is male', 'The person in the image is female']
    text_prompts_5 = ['The person in the image is him', 'The person in the image is her']
    text_prompts_6 = ['The person in the image is he', 'The person in the image is she']

    for text_prompts in [text_prompts_1, text_prompts_2, text_prompts_3, text_prompts_4, text_prompts_5, text_prompts_6]:
        print("\nText Prompt: ", text_prompts)
        print("---------------------------------------------------------------------")
        text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
        for key in text_inputs:
            text_inputs[key] = text_inputs[key].to(device)

        # Counters for evaluation
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
                # Get image and text features separately
                image_features = model.get_image_features(pixel_values=images_batch["pixel_values"])
                text_features = model.get_text_features(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"]
                )
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # Compute cosine similarity (logits)
                logits = image_features @ text_features.t()
                # For each image, the higher logit determines the predicted prompt:
                # index 0 corresponds to "a photo of a male" and index 1 to "a photo of a female"
                preds = torch.argmax(logits, dim=-1)
                for pred, true_gender in zip(preds, genders):
                    if true_gender == "male":
                        total_male += 1
                        if pred.item() == 0:
                            correct_male += 1
                    else:
                        total_female += 1
                        if pred.item() == 1:
                            correct_female += 1

        RA_m = correct_male / total_male if total_male > 0 else 0
        RA_f = correct_female / total_female if total_female > 0 else 0
        RA_avg = (RA_m + RA_f) / 2
        GG = abs(RA_m - RA_f)

        print(f"RA_m (Male Resolution Accuracy): {RA_m:.4f}")
        print(f"RA_f (Female Resolution Accuracy): {RA_f:.4f}")
        print(f"RA_avg (Average Resolution Accuracy): {RA_avg:.4f}")
        print(f"GG (Gender Gap): {GG:.4f}")

if __name__ == "__main__":
    main()
