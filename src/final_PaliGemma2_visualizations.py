import matplotlib.pyplot as plt

# Number of methods
n_methods = 5
# Create x-axis positions 0, 1, 2, 3, 4
x_positions = list(range(n_methods))
# Method names for labeling the ticks; note we use "paligemma2" as raw
method_labels = ["paligemma2", "vision_frozen", "text_frozen", "none_frozen", "projections_not_frozen"]

# Data for OO Results
# Data based on the provided CDA and DAUDoS results:
# CDA:
#  - paligemma2 (raw): GG = 0.3304
#  - vision_frozen: GG = 0.0261
#  - text_frozen: GG = 0.0261
#  - none_frozen: GG = 0.0522
#  - projections_not_frozen: GG = 0.0174
#
# DAUDoS:
#  - paligemma2 (raw): GG = 0.3304
#  - vision_frozen: GG = 0.0957
#  - text_frozen: GG = 0.2435
#  - none_frozen: GG = 0.0696
#  - projections_not_frozen: GG = 0.1826
data_OO = {
    "(PaliGemma2, CDA)": [0.3304, 0.0261, 0.0261, 0.0522, 0.0174],
    "(PaliGemma2, DAUDoS)": [0.3304, 0.0957, 0.2435, 0.0696, 0.1826],
}

# Data for OP Results
# CDA:
#  - paligemma2 (raw): GG = 0.4522
#  - vision_frozen: GG = 0.1261
#  - text_frozen: GG = 0.3870
#  - none_frozen: GG = 0.1609
#  - projections_not_frozen: GG = 0.0478
#
# DAUDoS:
#  - paligemma2 (raw): GG = 0.4522
#  - vision_frozen: GG = 0.3565
#  - text_frozen: GG = 0.4696
#  - none_frozen: GG = 0.4652
#  - projections_not_frozen: GG = 0.4609#  - projections_not_frozen: GG = 0.0478
#
# DAUDoS:
#  - paligemma2 (raw): GG = 0.4522
#  - vision_frozen: GG = 0.3565
data_OP = {
    "(PaliGemma2, CDA)": [0.4522, 0.1261, 0.3870, 0.1609, 0.0478],
    "(PaliGemma2, DAUDoS)": [0.4522, 0.3565, 0.4696, 0.4652, 0.4609],
}

# ----------------------------------
# Plot for OO Results using numerical x-axis with custom tick labels
# ----------------------------------
plt.figure(figsize=(10, 6))
for model, gg_values in data_OO.items():
    plt.plot(x_positions, gg_values, marker='o', label=model)
    
plt.xlabel("Method")
plt.ylabel("Gender Gap (GG)")
plt.title("OO Results: Gender Gap (GG) across Methods")
plt.xticks(ticks=x_positions, labels=method_labels)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../visualizations/PaliGemma2_OO_results.png')
# ----------------------------------
# Plot for OP Results using numerical x-axis with custom tick labels
# ----------------------------------
plt.figure(figsize=(10, 6))
for model, gg_values in data_OP.items():
    plt.plot(x_positions, gg_values, marker='o', label=model)
    
plt.xlabel("Method")
plt.ylabel("Gender Gap (GG)")
plt.title("OP Results: Gender Gap (GG) across Methods")
plt.xticks(ticks=x_positions, labels=method_labels)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('../visualizations/PaliGemma2_OP_results.png')
