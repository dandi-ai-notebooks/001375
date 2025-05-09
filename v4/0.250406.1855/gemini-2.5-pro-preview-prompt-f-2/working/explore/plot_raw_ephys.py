# Purpose: Load and plot a small segment of raw electrophysiology data
# from the NWB file to understand its basic characteristics.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use seaborn styling
sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # ensure read-only mode
nwb = io.read()

# Access time series data
time_series = nwb.acquisition["time_series"]
data = time_series.data
sampling_rate = time_series.rate  # Hz

# Parameters for plotting
duration_to_plot_sec = 0.1  # seconds
num_samples_to_plot = int(duration_to_plot_sec * sampling_rate)
channels_to_plot = [0, 1, 2]

# Get data subset
data_subset = data[:num_samples_to_plot, channels_to_plot]
time_vector = np.arange(num_samples_to_plot) / sampling_rate

# Create plot
fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(12, 2 * len(channels_to_plot)), sharex=True)
if len(channels_to_plot) == 1: # handle single channel case
    axes = [axes]

for i, channel_idx in enumerate(channels_to_plot):
    axes[i].plot(time_vector, data_subset[:, i])
    axes[i].set_ylabel(f'Channel {channel_idx}\n({time_series.unit})')
    axes[i].set_title(f'Raw Data - Channel {channel_idx}')

axes[-1].set_xlabel('Time (s)')
plt.suptitle('Raw Electrophysiology Data Snippet (First 0.1 seconds)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.savefig('explore/raw_ephys_snippet.png')
plt.close(fig) # Close the figure to free memory

io.close() # Close the NWBHDF5IO object
print("Plot saved to explore/raw_ephys_snippet.png")