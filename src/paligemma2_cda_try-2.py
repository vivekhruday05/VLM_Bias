import torch
from torch.utils.data import DataLoader
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset
import json
import os
from PIL import Image
import tqdm
import torch


# ----- Load and Prepare the Dataset -----
# Load your JSON file (assumed to be named "data.json")
with open("data/train_data.json", "r") as f:
    data_dict = json.load(f)

# Convert the dict into a list of samples, filtering out stereotypical samples
data_list = []
for image_name, info in data_dict.items():
    # Only include samples where stereotypical is False
    if info.get("stereotypical") is False:
        data_list.append({
            "image_name": image_name,
            "overall_caption": info["overall_caption"]
        })

dataset = Dataset.from_list(data_list)

model_id ="google/paligemma2-3b-pt-224" 
device = "cuda"

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
processor = PaliGemmaProcessor.from_pretrained(model_id)

image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
DTYPE = model.dtype

# module = input(
#     "Enter the modality to freeze (enter 'none' to not freeze anything), "
#     "enter 'text' to freeze text layers, 'vision' to freeze vision layers, "
#     "or 'projection' to only unfreeze projection layers): "
# )
modules = ["text", "vision", "projection", "none"]
for module in modules:
    for param in model.parameters():
        param.requires_grad = False
    if module == "text":
        print("Freezing text layers (language model), unfreezing vision & projection.")
        for param in model.vision_tower.parameters():
            param.requires_grad = True
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = True
    elif module == "vision":
        print("Freezing vision layers, unfreezing text & projection.")
        for param in model.language_model.parameters():
            param.requires_grad = True
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = True
    elif module == "projection":
        print("Freezing vision & text layers, unfreezing only projection.")
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = True
    elif module == "none":
        print("Not freezing anything")
        for param in model.parameters():
            param.requires_grad = True

    def collate_fn(examples):
        texts = ["<image> Describe the person in the image using correct gender pronouns" for _ in examples]
        labels = [example['overall_caption'] for example in examples]

        # Load images using their filenames
        images = [Image.open(os.path.join("img_align_celeba", example["image_name"])).convert("RGB") for example in examples]

        tokens = processor(text=texts, images=images, suffix=labels,
                        return_tensors="pt", padding="longest")

        tokens = tokens.to(DTYPE).to(device)
        return tokens

    args=TrainingArguments(
                num_train_epochs=3,
                remove_unused_columns=False,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=2,
                learning_rate=2e-5,
                weight_decay=1e-6,
                adam_beta2=0.999,
                logging_steps=100,
                optim="adamw_hf", # you can use paged optimizers like paged_adamw_8bit for QLoRA
                save_strategy="steps",
                save_steps=1000,
                save_total_limit=1,
                output_dir="paligemma_vqav2",
                bf16=True,
                dataloader_pin_memory=False
            )

    trainer = Trainer(
            model=model,
            train_dataset=dataset,
            data_collator=collate_fn,
            args=args
            )

    trainer.train()

    # Save the model
    if module:
        if module != "projection":
            model.save_pretrained(f"models/{module}_frozen")
            processor.save_pretrained(f"models/{module}_frozen")
        else:
            model.save_pretrained(f"models/projections_not_frozen")
            processor.save_pretrained(f"models/projections_not_frozen")
    else:
        model.save_pretrained(f"models/none_frozen")
        processor.save_pretrained(f"models/none_frozen")