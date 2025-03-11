import os
import json
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor, TrainingArguments, Trainer

# Custom Dataset for training
class CLIPTrainDataset(Dataset):
    def __init__(self, json_file, image_folder, processor):
        with open(json_file, "r") as f:
            data = json.load(f)
        self.samples = []
        for image_name, meta in data.items():
            # Filter only non-stereotypical male samples
            # if meta.get("stereotypical") == False and meta.get("gender") == "male":
            if meta.get("stereotypical") == False:
                image_path = os.path.join(image_folder, image_name)
                if os.path.exists(image_path):
                    caption = meta.get("overall_caption", "")
                    self.samples.append((image_path, caption))
        print(f"Found {len(self.samples)} training samples.")
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        # Force fixed length tokenization (CLIP uses 77 tokens)
        inputs = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )
        # Remove the extra batch dimension
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)
        return inputs

# Subclass Trainer to override compute_loss
class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, **kwargs):
        # Forward pass: the model does not return a loss by default.
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: (batch_size, batch_size)
        logits_per_text = outputs.logits_per_text
        batch_size = logits_per_image.size(0)
        labels = torch.arange(batch_size, device=logits_per_image.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        return loss

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_json = "../data/train_data.json"
    image_folder = "../data/img_align_celeba"

    # Load model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    # print(model)
    module = input("Enter the modality to freeze (Click enter to not freeze anything or enter 'projection' to only unfreeze it): ")

    if module != '':
    # Freeze the vision encoder so that its weights are not updated during fine-tuning.
        for param in model.parameters():
            param.requires_grad = False
        if module == "text":
            print("Freezing text")
            for param in model.vision_model.parameters():
                param.requires_grad = True
            for param in model.visual_projection.parameters():
                param.requires_grad = True
        elif module == "vision":
            print("Freezing vision")
            for param in model.text_model.parameters():
                param.requires_grad = True
            for param in model.text_projection.parameters():
                param.requires_grad = True
        elif module == "projection":
            for param in model.text_projection.parameters():
                param.requires_grad = True
            for param in model.visual_projection.parameters():
                param.requires_grad = True
    else:
        print("Not freezing anything")
        for param in model.parameters():
                param.requires_grad = True

    # Prepare training dataset
    train_dataset = CLIPTrainDataset(train_json, image_folder, processor)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./clip_finetuned",
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        learning_rate=1e-6,
        remove_unused_columns=False,  # Ensure all inputs are passed to the model
    )

    # Use our custom CLIPTrainer which overrides compute_loss
    trainer = CLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model and processor
    model.save_pretrained("fine_tuned_clip_hf")
    processor.save_pretrained("fine_tuned_clip_hf")
    print("Training complete. Model and processor saved in 'fine_tuned_clip_hf'.")

if __name__ == "__main__":
    main()
