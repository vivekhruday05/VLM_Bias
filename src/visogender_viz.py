import matplotlib.pyplot as plt
import numpy as np

# Define the methods (same for both settings)
methods = ["Raw Clip", "Vision Frozen", "Text Frozen", "None Frozen", "Only Projections Unfrozen"]
x = np.arange(len(methods))  # x-axis positions

# Data for OO (Occupation-Object) Results
RA_avg_oo = [0.9435, 0.9391, 0.9652, 0.9652, 0.9478]
GG_oo     = [0.0609, 0.0522, 0.0000, 0.0000, 0.0348]

# Data for OP (Occupation-Participant) Results
RA_avg_op = [0.5609, 0.5674, 0.5804, 0.6283, 0.5609]
GG_op     = [0.3043, 0.1696, 0.0826, 0.0652, 0.2696]

# Plot 1: RA_avg for OO and OP
plt.figure(figsize=(10, 6))
plt.plot(x, RA_avg_oo, marker='o', linestyle='-', label='RA_avg OO')
plt.plot(x, RA_avg_op, marker='s', linestyle='--', label='RA_avg OP')
plt.xticks(x, methods, rotation=45, ha='right')
plt.xlabel('Debiasing Method')
plt.ylabel('RA_avg')
plt.title('Comparison of RA_avg between OO and OP Settings')
plt.legend()
plt.tight_layout()
plt.savefig('../visualizations/visogender_ra_avg.png')  # Save the figure

# Plot 2: Gender Gap (GG) for OO and OP
plt.figure(figsize=(10, 6))
plt.plot(x, GG_oo, marker='o', linestyle='-', label='GG OO')
plt.plot(x, GG_op, marker='s', linestyle='--', label='GG OP')
plt.xticks(x, methods, rotation=45, ha='right')
plt.xlabel('Debiasing Method')
plt.ylabel('Gender Gap (GG)')
plt.title('Comparison of Gender Gap between OO and OP Settings')
plt.legend()
plt.tight_layout()
plt.savefig('../visualizations/visogender_gg.png')  # Save the figure
