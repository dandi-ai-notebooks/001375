"""
This script loads a subset of the time series data from the NWB file and visualizes it by plotting the first few time points for selected channels. The visualization is saved to a PNG file.

"""

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ce525828-8534-4b56-9e47-d2a34d1aa897/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract time series data
time_series = nwb.acquisition["time_series"]

# Load a subset of the data for visualization
num_samples = 1000  # Number of samples to load
data_subset = time_series.data[:num_samples, :5]  # Load first 5 channels

# Plot the data
plt.figure(figsize=(12, 6))
for i in range(data_subset.shape[1]):
    plt.plot(data_subset[:, i], label=f"Channel {i+1}")

plt.xlabel("Sample Index")
plt.ylabel("Amplitude (mV)")
plt.title("Time Series Data from First 5 Channels")
plt.legend()
plot_path = "explore/time_series_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

"""
Closing NWB file to ensure resources are freed
"""
io.close()