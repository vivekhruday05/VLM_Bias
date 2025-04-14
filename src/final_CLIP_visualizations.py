import matplotlib.pyplot as plt

# Number of methods
n_methods = 5
x_positions = list(range(n_methods))
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

# Create subplots for OO and OP in one figure
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharey=True)

# Plot OO Results
for model, gg_values in data_OO.items():
    axes[0].plot(x_positions, gg_values, marker='o', label=model)
axes[0].set_title("OO Results: Gender Gap (GG)")
axes[0].set_xlabel("Method")
axes[0].set_ylabel("Gender Gap (GG)")
axes[0].set_xticks(x_positions)
axes[0].set_xticklabels(method_labels, rotation=20)
axes[0].legend()
axes[0].grid(True)

# Plot OP Results
for model, gg_values in data_OP.items():
    axes[1].plot(x_positions, gg_values, marker='o', label=model)
axes[1].set_title("OP Results: Gender Gap (GG)")
axes[1].set_xlabel("Method")
axes[1].set_xticks(x_positions)
axes[1].set_xticklabels(method_labels, rotation=20)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("../visualizations/CLIP_OO_OP_results_subplot.png")
plt.show()
