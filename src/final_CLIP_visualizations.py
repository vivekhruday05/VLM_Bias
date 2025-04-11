import matplotlib.pyplot as plt

# Number of methods
n_methods = 5
# Create x-axis positions: 0, 1, 2, 3, 4
x_positions = list(range(n_methods))
# Method names for labeling the ticks; we use "clip" as the raw method.
method_labels = ["clip", "vision_frozen", "text_frozen", "none_frozen", "projections_unfrozen"]

# ----------------------------------------------------------------
# Data for CLIP results (using "clip" as the raw method)
# ----------------------------------------------------------------

# CLIP, CDA:
#   - clip:          OO GG = 0.0609, OP GG = 0.3043
#   - vision_frozen: OO GG = 0.0522, OP GG = 0.1696
#   - text_frozen:   OO GG = 0.0000, OP GG = 0.0826
#   - none_frozen:   OO GG = 0.0000, OP GG = 0.0652
#   - projections...:OO GG = 0.0348, OP GG = 0.2696
#
# CLIP, Task Vector:
#   - clip:          OO GG = 0.0609, OP GG = 0.3043
#   - vision_frozen: OO GG = 0.5739, OP GG = 0.0783
#   - text_frozen:   OO GG = 0.3913, OP GG = 0.3348
#   - none_frozen:   OO GG = 0.1913, OP GG = 0.2870
#   - projections...:OO GG = 0.3913, OP GG = 0.3348
#
# CLIP, DAUDoS:
#   - clip:          OO GG = 0.0609, OP GG = 0.3043
#   - vision_frozen: OO GG = 0.0696, OP GG = 0.3696
#   - text_frozen:   OO GG = 0.0348, OP GG = 0.2870
#   - none_frozen:   OO GG = 0.0522, OP GG = 0.3435
#   - projections...:OO GG = 0.0696, OP GG = 0.3130

data_OO = {
    "(CLIP, CDA)": [0.0609, 0.0522, 0.0000, 0.0000, 0.0348],
    "(CLIP, DAUDoS)": [0.0609, 0.0696, 0.0348, 0.0522, 0.0696],
}

data_OP = {
    "(CLIP, CDA)": [0.3043, 0.1696, 0.0826, 0.0652, 0.2696],
    "(CLIP, DAUDoS)": [0.3043, 0.3696, 0.2870, 0.3435, 0.3130],
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
plt.savefig('../visualizations/CLIP_OO_results.png')

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
plt.savefig('../visualizations/CLIP_OP_results.png')
