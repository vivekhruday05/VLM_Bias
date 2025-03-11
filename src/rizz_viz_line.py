import matplotlib.pyplot as plt

# Define finetuning settings
settings = [
    "Raw CLIP", 
    "Vision Frozen", 
    "Text Frozen",
    "None Frozen",
    "Everything Except Projections Frozen"
]

# Define prompts
prompts = [
    "['male', 'female']",
    "['him', 'her']",
    "['he', 'she']",
    "['The person in the image is male', 'The person in the image is female']",
    "['The person in the image is him', 'The person in the image is her']",
    "['The pronoun of the person in the image is he', 'The pronoun of the person in the image is she']"
]

# Metrics data (values for each setting across prompts)
RA_m_data = [
    [0.9930, 0.9859, 0.9836, 0.5798, 0.9859, 0.9836],  # Raw CLIP
    [0.9883, 0.9859, 0.9812, 0.9789, 0.9859, 0.9789],  # Vision Frozen
    [1.0000, 0.9930, 0.9906, 0.7535, 0.9930, 0.9906],  # Text Frozen
    [0.9953, 0.9930, 0.9906, 0.9859, 0.9906, 0.9906],  # None Frozen
    [0.9930, 0.9859, 0.9836, 0.6479, 0.9836, 0.9789],  # Everything Except Projections Frozen
]

RA_f_data = [
    [0.9126, 0.9874, 0.9784, 0.9856, 0.9883, 0.9892],  # Raw CLIP
    [0.9676, 0.9856, 0.9883, 0.9892, 0.9883, 0.9892],  # Vision Frozen
    [0.4811, 0.9225, 0.9766, 0.9757, 0.9586, 0.9811],  # Text Frozen
    [0.9072, 0.9532, 0.9766, 0.9811, 0.9730, 0.9829],  # None Frozen
    [0.8252, 0.9838, 0.9712, 0.9486, 0.9856, 0.9892],  # Everything Except Projections Frozen
]

RA_avg_data = [
    [0.9528, 0.9867, 0.9810, 0.7827, 0.9871, 0.9864],  # Raw CLIP
    [0.9513, 0.9731, 0.9836, 0.9835, 0.9818, 0.9867],  # None Fro
    [0.9779, 0.9858, 0.9848, 0.9840, 0.9871, 0.9840],  # Vision Frozen
    [0.7405, 0.9577, 0.9836, 0.8646, 0.9758, 0.9858],  # Text Frozen
    [0.9072, 0.9532, 0.9766, 0.9811, 0.9730, 0.9829],  # None Frozen
    [0.9091, 0.9848, 0.9785, 0.8605, 0.9869, 0.9852],  # Everything Except Projections Frozen
]

GG_data = [
    [0.0803, 0.0015, 0.0052, 0.4058, 0.0024, 0.0056],  # Raw CLIP
    [0.0207, 0.0003, 0.0071, 0.0103, 0.0024, 0.0103],  # Vision Frozen
    [0.5189, 0.0704, 0.0140, 0.2222, 0.0344, 0.0095],  # Text Frozen
    [0.0881, 0.0398, 0.0140, 0.0048, 0.0176, 0.0077],  # None Frozen
    [0.1677, 0.0021, 0.0147, 0.1763, 0.0027, 0.0080],  # Everything Except Projections Frozen
]

# Define a list of colors for each prompt
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

def plot_metric_lines_all(metric_data, metric_name):
    plt.figure(figsize=(12, 8))
    x_values = list(range(len(settings)))  # Numeric indices for the x-axis
    # Plot a line for each prompt
    for i, prompt in enumerate(prompts):
        # Extract values for this prompt across all finetuning settings
        values = [metric_data[j][i] for j in range(len(settings))]
        plt.plot(x_values, values, marker='o', color=colors[i], label=prompt, linestyle='-')
    
    plt.title(f"{metric_name} Across Finetuning Settings", fontsize=16)
    plt.ylabel(metric_name)
    plt.xticks(x_values, settings, rotation=15, fontsize=8)
    plt.ylim(0, 0.6)
    plt.legend(loc="upper right", fontsize=6)
    plt.tight_layout()
    plt.savefig(f'../visualizations/results/{metric_name}.jpg')

# Plot each metric in a single figure with all prompt lines
# plot_metric_lines_all(RA_m_data, "RA_m (Male Accuracy)")
# plot_metric_lines_all(RA_f_data, "RA_f (Female Accuracy)")
# plot_metric_lines_all(RA_avg_data, "RA_avg (Average Accuracy)")
plot_metric_lines_all(GG_data, "GG (Gender Gap)")
