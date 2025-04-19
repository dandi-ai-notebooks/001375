# %%
# This script plots the extracellular electrophysiology data from the first 10 channels.
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the extracellular electrophysiology data
data = nwb.acquisition["time_series"].data
rate = nwb.acquisition["time_series"].rate

# Plot the first 10 channels for the first 1 second of data
num_channels = min(10, data.shape[1])
duration = 1  # seconds
num_timepoints = int(duration * rate)
fig, axes = plt.subplots(num_channels, 1, figsize=(10, num_channels * 2))
for i in range(num_channels):
    channel_data = data[:num_timepoints, i]
    time = np.arange(num_timepoints) / rate
    axes[i].plot(time, channel_data)
    axes[i].set_ylabel(f"Channel {i}")
axes[0].set_title("Extracellular Electrophysiology Data (First 10 Channels)")
axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("explore/ephys_data.png")