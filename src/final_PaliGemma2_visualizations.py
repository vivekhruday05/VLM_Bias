import matplotlib.pyplot as plt
import numpy as np

# Number of methods and x-axis positions
n_methods = 5
x_positions = np.arange(n_methods)
method_labels = ["paligemma2", "vision_frozen", "text_frozen", "none_frozen", "projections_not_frozen"]

# OO Results
data_OO = {
    "(PaliGemma2, CDA)": [0.3304, 0.0261, 0.0261, 0.0522, 0.0174],
    "(PaliGemma2, DAUDoS)": [0.3304, 0.0957, 0.2435, 0.0696, 0.1826],
}

# OP Results
data_OP = {
    "(PaliGemma2, CDA)": [0.4522, 0.1261, 0.3870, 0.1609, 0.0478],
    "(PaliGemma2, DAUDoS)": [0.4522, 0.3565, 0.4696, 0.4652, 0.4609],
}

bar_width = 0.35

# Plotting OO results
plt.figure(figsize=(10, 5))
plt.bar(x_positions - bar_width/2, data_OO["(PaliGemma2, CDA)"], width=bar_width, label="CDA")
plt.bar(x_positions + bar_width/2, data_OO["(PaliGemma2, DAUDoS)"], width=bar_width, label="DAUDoS")
plt.xticks(x_positions, method_labels, rotation=45)
plt.title("OO Results")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig('../visualizations/PaliGemma2_OO.png')


# Plotting OP results
plt.figure(figsize=(10, 5))
plt.bar(x_positions - bar_width/2, data_OP["(PaliGemma2, CDA)"], width=bar_width, label="CDA")
plt.bar(x_positions + bar_width/2, data_OP["(PaliGemma2, DAUDoS)"], width=bar_width, label="DAUDoS")
plt.xticks(x_positions, method_labels, rotation=45)
plt.title("OP Results")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig('../visualizations/PaliGemma2_OP.png')

