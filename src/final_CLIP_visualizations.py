import matplotlib.pyplot as plt
import numpy as np

# Number of methods
n_methods = 5
x_positions = np.arange(n_methods)
method_labels = ["clip", "vision_frozen", "text_frozen", "none_frozen", "projections_unfrozen"]

# Data for CLIP results
data_OO = {
    "(CLIP, CDA)": [0.0609, 0.0522, 0.0000, 0.0000, 0.0348],
    "(CLIP, DAUDoS)": [0.0609, 0.0696, 0.0348, 0.0522, 0.0696],
}

data_OP = {
    "(CLIP, CDA)": [0.3043, 0.1696, 0.0826, 0.0652, 0.2696],
    "(CLIP, DAUDoS)": [0.3043, 0.3696, 0.2870, 0.3435, 0.3130],
}

bar_width = 0.35

# First plot: OO setting
plt.figure(figsize=(10, 5))
plt.bar(x_positions - bar_width/2, data_OO["(CLIP, CDA)"], width=bar_width, label="(CLIP, CDA)")
plt.bar(x_positions + bar_width/2, data_OO["(CLIP, DAUDoS)"], width=bar_width, label="(CLIP, DAUDoS)")
plt.title("OO Setting")
plt.xticks(x_positions, method_labels, rotation=15)
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig('../visualizations/CLIP_OO.png')

# Second plot: OP setting
plt.figure(figsize=(10, 5))
plt.bar(x_positions - bar_width/2, data_OP["(CLIP, CDA)"], width=bar_width, label="(CLIP, CDA)")
plt.bar(x_positions + bar_width/2, data_OP["(CLIP, DAUDoS)"], width=bar_width, label="(CLIP, DAUDoS)")
plt.title("OP Setting")
plt.xticks(x_positions, method_labels, rotation=15)
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig('../visualizations/CLIP_OP.png')