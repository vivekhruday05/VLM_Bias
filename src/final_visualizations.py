import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data and settings
method_labels_clip = ["clip", "vision\nfrozen", "text\nfrozen", "none\nfrozen", "projections\nunfrozen"]
method_labels_pali = ["paligemma2", "vision\nfrozen", "text\nfrozen", "none\nfrozen", "projections\nunfrozen"]
x = np.arange(len(method_labels_clip))
bar_width = 0.35

# CLIP results
data_clip_OO = [0.0609, 0.0522, 0.0000, 0.0000, 0.0348]
data_clip_OP = [0.3043, 0.1696, 0.0826, 0.0652, 0.2696]
data_clip_OO_dau = [0.0609, 0.0696, 0.0348, 0.0522, 0.0696]
data_clip_OP_dau = [0.3043, 0.3696, 0.2870, 0.3435, 0.3130]

# PaliGemma2 results
data_pali_OO = [0.3304, 0.0261, 0.0261, 0.0522, 0.0174]
data_pali_OO_dau = [0.3304, 0.0957, 0.2435, 0.0696, 0.1826]
data_pali_OP = [0.4522, 0.1261, 0.3870, 0.1609, 0.0478]
data_pali_OP_dau = [0.4522, 0.3565, 0.4696, 0.4652, 0.4609]

# Seaborn style and palette
sns.set_style("whitegrid")
palette = sns.color_palette("muted", 2)

# Create subplots with extra vertical space between rows
fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'hspace': 0.35})
axes = axes.flatten()

# Helper to plot bars and annotate values
def plot_bars(ax, x, data1, data2, labels, title):
    bars1 = ax.bar(x - bar_width/2, data1, width=bar_width, label='CDA', color=palette[0])
    bars2 = ax.bar(x + bar_width/2, data2, width=bar_width, label='DAUDoS', color=palette[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')  # Smaller title font size
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    # Annotate values just above each bar with minimal gap
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )

# Plotting each subplot
plot_bars(axes[0], x, data_clip_OO, data_clip_OO_dau, method_labels_clip, 'GG - CLIP - OO Setting')
plot_bars(axes[1], x, data_clip_OP, data_clip_OP_dau, method_labels_clip, 'GG - CLIP - OP Setting')
plot_bars(axes[2], x, data_pali_OO, data_pali_OO_dau, method_labels_pali, 'GG - PaliGemma2 - OO Setting')
plot_bars(axes[3], x, data_pali_OP, data_pali_OP_dau, method_labels_pali, 'GG - PaliGemma2 - OP Setting')

# Add common legend at the top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=13, frameon=False)

# Layout adjustments
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('../visualizations/final_visualizations.png', dpi=300)
