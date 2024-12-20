import numpy as np
import matplotlib.pyplot as plt


# Define time variable
t = np.linspace(0, 10, 500)

# Exponential decay for half-life
alpha = 0.8
half_life_decay = np.exp(-alpha * t)
half_life_time = np.log(2) / alpha  # Calculate half-life time based on decay rate

# Parameters for concentration profile
a = 0.5
k = 1.2
concentration_profile = 1 / (k - a) * (np.exp(-a * t) - np.exp(-k * t))
peak_time = ( np.log(a) - np.log(k))/(a - k)  # Given approximate peak time


# Set font sizes for plots
font_size_labels = 14
font_size_ticks = 14
font_size_legend = 14

# First plot: Exponential decay for half-life with marked half-life time as t_{1/2}
plt.figure(figsize=(8, 5))
plt.plot(t, half_life_decay, color="blue", label = 'Antibiotics concentration')
plt.axhline(0.5, color="gray", linestyle="--", label="Half-life")
plt.axvline(half_life_time, color="black", linestyle="--")  # Vertical line for half-life time
plt.xticks([0, 2, 4, 6, 8, 10, half_life_time], labels=[0, 2, 4, 6, 8, 10, r"$t_{1/2}$"],
           fontsize = font_size_ticks)  # Set ticks from 0 to 10 with t_{1/2}
plt.yticks(fontsize=font_size_ticks)
plt.xlim(0, 10)  # Limit x-axis from 0 to 10

plt.xlabel("Time", fontsize = font_size_labels)
plt.ylabel("Concentration", fontsize = font_size_labels)
plt.legend(fontsize = font_size_legend)
plt.grid(True)


# Second plot: Concentration profile with MIC, AUC, and marked peak time as t_{peak}
plt.figure(figsize=(8, 5))
plt.plot(t, concentration_profile, color="red", label = 'Antibiotics in bloodstream')
# Fill area under the curve for AUC
plt.fill_between(t, concentration_profile, where=(t <= 10), color="blue", alpha=0.2, label="AUC")
# Horizontal line for MIC
plt.axhline(0.2, color="gray", linestyle="--", label="MIC")
# Vertical line for peak time
plt.axvline(peak_time, color="black", linestyle="--")
plt.xticks([0, 2, 4, 6, 8, 10, peak_time], labels=[0, 2, 4, 6, 8, 10, r"$t_{peak}$"],
           fontsize = font_size_ticks)  # Set ticks from 0 to 10 with t_{peak}
plt.yticks(fontsize = font_size_ticks)
plt.xlim(0, 10)  # Limit x-axis from 0 to 10

plt.xlabel("Time", fontsize = font_size_labels)
plt.ylabel("Concentration", fontsize = font_size_labels)
plt.legend(fontsize = font_size_legend)
plt.grid(True)
plt.show()