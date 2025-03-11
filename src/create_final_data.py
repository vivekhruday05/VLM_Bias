import json
import random
from sklearn.model_selection import train_test_split

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def clean_captions(captions):
    for k, v in captions.items():
        if "attribute_wise_captions" in v:
            del v["attribute_wise_captions"]

def process_dataset(captions_file):
    captions = load_json(captions_file)
    
    true_examples = {}
    false_examples = {}
    
    clean_captions(captions)

    additional_true = {k: v for k, v in captions.items() if v["stereotypical"]}
    additional_false = {k: v for k, v in captions.items() if not v["stereotypical"]}
    
    true_examples.update(dict(list(additional_true.items())))
    true_count = len(true_examples)

    false_count_needed = true_count - len(false_examples)
    additional_false_items = list(additional_false.items())
    random.shuffle(additional_false_items)
    false_examples.update(dict(additional_false_items[:false_count_needed]))
    
    final_data = {**true_examples, **false_examples}
    return final_data

final_data = process_dataset("../data/captions_with_stereotypes.json")

# Split data into train, test, and validation sets
keys = list(final_data.keys())
train_keys, test_keys = train_test_split(keys, test_size=0.2, random_state=42)
test_keys, val_keys = train_test_split(test_keys, test_size=0.5, random_state=42)

train_data = {k: final_data[k] for k in train_keys}
test_data = {k: final_data[k] for k in test_keys}
val_data = {k: final_data[k] for k in val_keys}

# Save the splits to JSON files
with open("../data/train_data.json", "w") as f:
    json.dump(train_data, f)

with open("../data/test_data.json", "w") as f:
    json.dump(test_data, f)

with open("../data/val_data.json", "w") as f:
    json.dump(val_data, f)

