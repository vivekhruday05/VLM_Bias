import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import random

def sample_gender_data(json_file, sample_size=50000):
    with open(json_file, "r") as f:
        captions = json.load(f)

    # Separate male and female entries
    males = {k: v for k, v in captions.items() if v.get("gender") == "male"}
    females = {k: v for k, v in captions.items() if v.get("gender") == "female"}

    # Sample 50k from each
    males_sample = dict(random.sample(list(males.items()), min(sample_size, len(males))))
    females_sample = dict(random.sample(list(females.items()), min(sample_size, len(females))))

    # Combine sampled data
    sampled_data = {**males_sample, **females_sample}

    return sampled_data

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

def plot_gender_attribute_distribution(json_file, annotation_file):

    captions = sample_gender_data(json_file)
    annotations = load_txt(annotation_file)

    data = []
    for img_name, details in captions.items():
        if img_name in annotations:
            row = annotations[img_name]
            row["gender"] = details.get("gender", "unknown")
            row["img_name"] = img_name
            data.append(row)

    df = pd.DataFrame(data)

    attributes = ["Bangs", "Eyeglasses", "No_Beard", "Smiling", "Young"]
    colors = {"male": "blue", "female": "pink", "unknown": "orange"}

    for attr in attributes:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()

        unique_values = sorted(df[attr].unique())  # Unique values of the attribute
        positions = np.arange(len(unique_values))  # Correct x positions

        width = 0.4  # Width of bars
        for i, (gender, color) in enumerate(colors.items()):
            subset = df[df["gender"] == gender]
            counts = subset[attr].value_counts().reindex(unique_values, fill_value=0)  # Ensure all values are present
            x = positions + (i * width)  # Shift bars

            ax.bar(x, counts, width=width, label=gender, color=color)

        plt.xlabel(attr)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {attr} by Gender")
        plt.xticks(positions + width, unique_values)  # Align x labels
        plt.legend()
        plt.savefig("../visualizations/data/attribute_distribution_" + attr + ".png")

plot_gender_attribute_distribution("../data/captions_with_gender.json", "../data/combined_annotation.txt")

