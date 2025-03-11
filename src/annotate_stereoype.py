import json

def is_stereotypical(attributes, gender):
    return (
        (gender == "female" and
        attributes["Bangs"] >= 2 and
        attributes["Eyeglasses"] <= 1 and
        attributes["No_Beard"] == 0 and
        attributes["Smiling"] >= 3 and
        attributes["Young"] <= 2)
        or
        (gender == "male" and
        attributes["Bangs"] < 2 and
        attributes["Eyeglasses"] > 1 and
        attributes["No_Beard"] != 0 and
        attributes["Smiling"] < 3 and
        attributes["Young"] > 2)
    )

# Load captions.json
def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def load_txt(filename):
    data = {}
    with open(filename, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            img_name = parts[0]
            attributes = list(map(int, parts[1:]))
            data[img_name] = {
                "Bangs": attributes[0],
                "Eyeglasses": attributes[1],
                "No_Beard": attributes[2],
                "Smiling": attributes[3],
                "Young": attributes[4]
            }
    return data

captions = load_json("../data/captions_with_gender.json")
annotations = load_txt("../data/combined_annotation.txt")

for img, attr in annotations.items():
    if img in captions:
        captions[img]["stereotypical"] = is_stereotypical(attr, captions[img]["gender"])

with open("../data/captions_with_stereotypes.json", "w") as f:
    json.dump(captions, f, indent=4)

print("Stereotypical attribute added successfully.")
