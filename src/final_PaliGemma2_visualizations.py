import matplotlib.pyplot as plt

# Number of methods and x-axis positions
n_methods = 5
x_positions = list(range(n_methods))
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

# Create subplots: 1 row, 2 columns
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# --- Subplot 1: OO Results ---
for model, gg_values in data_OO.items():
    axs[0].plot(x_positions, gg_values, marker='o', label=model)

axs[0].set_title("OO Results: Gender Gap (GG) across Methods")
axs[0].set_xlabel("Method")
axs[0].set_ylabel("Gender Gap (GG)")
axs[0].set_xticks(x_positions)
axs[0].set_xticklabels(method_labels, rotation=30)
axs[0].grid(True)
axs[0].legend()

# --- Subplot 2: OP Results ---
for model, gg_values in data_OP.items():
    axs[1].plot(x_positions, gg_values, marker='o', label=model)

axs[1].set_title("OP Results: Gender Gap (GG) across Methods")
axs[1].set_xlabel("Method")
axs[1].set_xticks(x_positions)
axs[1].set_xticklabels(method_labels, rotation=30)
axs[1].grid(True)
axs[1].legend()

# Layout adjustment and save
plt.tight_layout()
plt.savefig('../visualizations/PaliGemma2_OO_OP_results_subplot.png')
plt.show()
