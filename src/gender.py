import json

def detect_gender(text):
    male_pronouns = {"he", "his", "him", "man", "boy", "gentleman", "male"}
    female_pronouns = {"she", "her", "woman", "girl", "lady", "female"}
    
    words = set(text.lower().split())
    if words & female_pronouns:
        return "female"
    elif words & male_pronouns:
        return "male"
    return "unknown"

with open("captions.json", "r") as f:
    data = json.load(f)

for key, value in data.items():
    for key_1, value_1 in value["attribute_wise_captions"].items():
        if value_1 is None:
            value["attribute_wise_captions"][key_1] = "None"

for key, value in data.items():
    combined_text = " ".join(value["attribute_wise_captions"].values()) + " " + value["overall_caption"]
    gender = detect_gender(combined_text)
    data[key]["gender"] = gender

with open("captions_with_gender.json", "w") as f:
    json.dump(data, f, indent=4)

print("Gender attribute added successfully.")
